import torch
import math
import csv
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributed as dist
from collections import Counter
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaModel, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device('cuda')


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_heads, vocab_size, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.d_k = emb_dim // num_heads

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.qkv_layer = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.final_linear = nn.Linear(self.emb_dim, vocab_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, emb_dim)
        )
        self.enc_norm1 = nn.LayerNorm(self.emb_dim)
        self.enc_norm2 = nn.LayerNorm(self.emb_dim)
        self.dec_norm1 = nn.LayerNorm(self.emb_dim)
        self.dec_norm2 = nn.LayerNorm(self.emb_dim)
        self.dec_norm3 = nn.LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc, dec, enc_mask, dec_mask, enc_num, dec_num):
        # variables for both encoder and decoder
        batch_size = enc.shape[0]

        # embedding layer, comment it when using pretrained embedding models
        enc = self.embedding(enc)
        dec = self.embedding(dec)

        # positional embedding
        enc = sinusoidal_encoding(enc)
        dec = sinusoidal_encoding(dec)

        # variables for encoder matrix
        enc_seq_length = enc.shape[1]
        enc_mask = enc_mask.unsqueeze(1).unsqueeze(2)

        # variables for decoder matrix
        dec_seq_length = dec.shape[1]
        subsequent_mask = torch.triu(torch.ones((dec_seq_length, dec_seq_length),
                                                device=device), diagonal=1).float()
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 1, float('-inf'))

        combined_mask = dec_mask.unsqueeze(1).unsqueeze(2) + subsequent_mask.unsqueeze(0).unsqueeze(0)

        # loop for encoders
        for _ in range(enc_num):
            qkv = self.qkv_layer(enc)
            qkv = qkv.view(batch_size, enc_seq_length, self.num_heads, 3, self.d_k)
            qkv = qkv.permute(0, 2, 3, 1, 4).contiguous()
            enc_q, enc_k, enc_v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            enc_k_t = enc_k.transpose(-2, -1)

            enc_attn_score = torch.matmul(enc_q, enc_k_t)
            enc_attn_score = enc_attn_score + enc_mask
            enc_attn_weights = nn.functional.softmax(enc_attn_score
                                                     / math.sqrt(self.d_k),
                                                     dim=-1)
            enc_context = torch.matmul(enc_attn_weights, enc_v)

            # concatenate all the heads and apply linear layer
            enc_context = enc_context.transpose(1, 2)
            enc_context = enc_context.reshape(batch_size, enc_seq_length,
                                              self.emb_dim)
            enc_context = self.linear(enc_context)

            # apply add & norm layers and linear layers with skip connection
            enc = self.enc_norm1(enc + self.dropout(enc_context))
            ff_output = self.feed_forward(enc)
            enc = self.enc_norm2(enc + self.dropout(ff_output))

        # loop for decoders
        for _ in range(dec_num):
            dec_qkv = self.qkv_layer(dec)
            dec_qkv = dec_qkv.view(batch_size, dec_seq_length, self.num_heads, 3, self.d_k)
            dec_qkv = dec_qkv.permute(0, 2, 3, 1, 4).contiguous()
            dec_q, dec_k, dec_v = dec_qkv[:, :, 0], dec_qkv[:, :, 1], dec_qkv[:, :, 2]
            dec_k_t = dec_k.transpose(-2, -1)

            dec_attn_score = torch.matmul(dec_q, dec_k_t)
            dec_attn_score = dec_attn_score + combined_mask
            dec_attn_weights = nn.functional.softmax(dec_attn_score
                                                     / math.sqrt(self.d_k),
                                                     dim=-1)
            dec_context = torch.matmul(dec_attn_weights, dec_v)

            # concatenate all the heads and apply linear layer
            dec_context = dec_context.transpose(1, 2)
            dec_context = dec_context.reshape(batch_size, dec_seq_length,
                                              self.emb_dim)
            dec_context = self.linear(dec_context)

            # add and norm
            dec = self.dec_norm1(dec + self.dropout(dec_context))

            # calculate cross attention
            dec_q = self.qkv_layer(dec)[:, :, :self.emb_dim].view(batch_size,
                                                                  dec_seq_length, self.num_heads,
                                                                  self.d_k).permute(0, 2, 1, 3)
            enc_kv = self.qkv_layer(enc)
            enc_kv = enc_kv.view(batch_size, enc_seq_length, self.num_heads, 3, self.d_k)
            enc_kv = enc_kv.permute(0, 2, 3, 1, 4).contiguous()
            enc_k, enc_v = enc_kv[:, :, 1], enc_kv[:, :, 2]
            enc_k_t = enc_k.transpose(-2, -1)

            cross_attn_score = torch.matmul(dec_q, enc_k_t)
            cross_attn_score = cross_attn_score + enc_mask
            cross_attn_weights = nn.functional.softmax(cross_attn_score
                                                       / math.sqrt(self.d_k),
                                                       dim=-1)
            cross_context = torch.matmul(cross_attn_weights, enc_v)

            # concatenate all the heads and apply linear layer
            cross_context = cross_context.transpose(1, 2)
            cross_context = cross_context.reshape(batch_size, dec_seq_length,
                                                  self.emb_dim)
            cross_context = self.linear(cross_context)

            # add and norm
            dec = self.dec_norm2(dec + self.dropout(cross_context))

            # put it through feed forward and add and norm
            ff_output = self.feed_forward(dec)
            dec = self.dec_norm3(dec + self.dropout(ff_output))

        # finally, the linear layer with the softmax layer for output probabilities
        output = self.final_linear(dec)
        output = nn.functional.softmax(output, dim=-1)

        return output


class CustomDataset(Dataset):
    def __init__(self, enc, dec, enc_mask, dec_mask, dec_ids):
        self.enc = enc
        self.enc_mask = enc_mask
        self.dec = dec
        self.dec_mask = dec_mask
        self.dec_ids = dec_ids

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return {
            'enc': self.enc[idx],
            'enc_mask': self.enc_mask[idx],
            'dec': self.dec[idx],
            'dec_mask': self.dec_mask[idx],
            'dec_ids': self.dec_ids[idx],
        }


class CustomDataset2(Dataset):
    def __init__(self, enc_ids, dec_ids, enc_mask, dec_mask):
        self.enc_ids = enc_ids
        self.dec_ids = dec_ids
        self.enc_mask = enc_mask
        self.dec_mask = dec_mask

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        return {
            'enc': self.enc_ids[idx],
            'enc_mask': self.enc_mask[idx],
            'dec': self.dec_ids[idx],
            'dec_mask': self.dec_mask[idx],
        }


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_encoder_layers, num_decoder_layers, ff_dim, max_seq_length):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def create_positional_encoding(self, max_seq_length, emb_dim):
        pos_encoding = torch.zeros(max_seq_length, emb_dim)
        for pos in range(max_seq_length):
            for i in range(0, emb_dim, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_dim)))
                if i + 1 < emb_dim:
                    pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / emb_dim)))
        return pos_encoding.unsqueeze(0)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embedding and positional encoding
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Encoder and decoder
        memory = self.encoder(src_emb.permute(1, 0, 2), src_key_padding_mask=src_mask)
        output = self.decoder(tgt_emb.permute(1, 0, 2), memory, tgt_key_padding_mask=tgt_mask,
                              memory_key_padding_mask=src_mask)

        # Final linear layer
        output = self.fc_out(output.permute(1, 0, 2))
        return output


def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf values.")


def generate_embedding(corpus, batch_size, max_length):
    '''
    Takes in a corpus and returns an embedding matrix and the mask matrix.
    '''

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

    # put model on GPU and use FP16
    model.to(device)
    model.half()

    corpus_embeddings = []
    masks = []
    input_ids = []

    tokens = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    for i in range(0, len(corpus), batch_size):
        batch_ids = tokens['input_ids'][i:i + batch_size]
        batch_masks = tokens['attention_mask'][i:i + batch_size]
        batch_input = {'input_ids': batch_ids, 'attention_mask': batch_masks}

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(**batch_input)

        corpus_embeddings.append(output.last_hidden_state)
        masks.append(batch_masks)
        input_ids.append(batch_ids)

    embeddings = torch.cat(corpus_embeddings, dim=0)
    masks = torch.cat(masks, dim=0).half()

    # change masks so paddings are converted to -inf values
    masks[masks == 0] = float('-inf')
    masks[masks == 1] = 0

    return embeddings, masks, input_ids


def sinusoidal_encoding(embedding):
    '''
    Given an embedding of a sentence, calculates and returns the
    positional embedding matrix added to the embedding
    '''

    seq_length = embedding.shape[1]
    embedding_dim = embedding.shape[2]

    pos_encoding = []

    # calculate positional encoding
    for i in range(seq_length):
        word_encoding = []
        for j in range(embedding_dim):
            if j % 2 == 0:
                pe = math.sin(i / 10000 ** (2 * j / embedding_dim))
            else:
                pe = math.cos(i / 10000 ** (2 * j / embedding_dim))
            word_encoding.append(pe)
        pos_encoding.append(word_encoding)

    # convert encoding to torch
    pos_torch = torch.tensor(pos_encoding, device=device)

    return torch.add(embedding, pos_torch.unsqueeze(0))


def save_embeddings():
    question = []
    answer = []

    with open('Conversation.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question.append(row[1])
            answer.append(row[2])

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    max_length = tokenizer.model_max_length

    q_tokens = [tokenizer.tokenize(sentence)[:max_length] for sentence in question]
    a_tokens = [tokenizer.tokenize(sentence)[:max_length] for sentence in answer]

    q_token_counts = [len(tokens) for tokens in q_tokens]
    a_token_counts = [len(tokens) for tokens in a_tokens]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax[0].hist(q_token_counts, bins=range(1, max(q_token_counts) + 1), edgecolor='blue')
    ax[0].set_title("Human's Token Length Distribution")
    ax[0].set_xlabel("Token Length")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(a_token_counts, bins=range(1, max(a_token_counts) + 1), edgecolor='green')
    ax[1].set_title("AI's Word Length Distribution")
    ax[1].set_xlabel("Word Length")

    plt.show()

    q_emb, q_masks, q_ids = generate_embedding(question, 32, 20)
    q_emb = sinusoidal_encoding(q_emb)
    torch.save(q_emb, 'q_emb.pt')
    torch.save(q_masks, 'q_masks.pt')

    a_emb, a_masks, a_ids = generate_embedding(answer, 32, 20)
    a_emb = sinusoidal_encoding(a_emb)
    torch.save(a_emb, 'a_emb.pt')
    torch.save(a_masks, 'a_masks.pt')
    torch.save(a_ids, 'a_ids.pt')

    return


def save_letter_embeddings(max_length):
    question = []
    answer = []

    with open('Conversation.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question.append(row[1])
            answer.append(row[2])

    # 43 is for EOS, 44 for padding tokens
    valid_chars = (" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.’'\"“”!?"
                   "0123456789—_[]:;\n")
    char_to_int = {char: idx for idx, char in enumerate(valid_chars)}

    q_ids, q_masks = convert_to_integers(question, max_length, char_to_int)
    a_ids, a_masks = convert_to_integers(answer, max_length, char_to_int)

    q_token_counts = [len(tokens) for tokens in question]
    a_token_counts = [len(tokens) for tokens in answer]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax[0].hist(q_token_counts, bins=range(1, max(q_token_counts) + 1), edgecolor='blue')
    ax[0].set_title("Human's Token Length Distribution")
    ax[0].set_xlabel("Token Length")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(a_token_counts, bins=range(1, max(a_token_counts) + 1), edgecolor='green')
    ax[1].set_title("AI's Word Length Distribution")
    ax[1].set_xlabel("Word Length")

    plt.show()

    torch.save(q_ids, 'q_ids.pt')
    torch.save(q_masks, 'q_masks.pt')
    torch.save(a_ids, 'a_ids.pt')
    torch.save(a_masks, 'a_masks.pt')


def save_corpus_emb_ids(corpus, file_name):
    '''
    Saves input ids from an entire corpus.
    '''
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    tokens = tokenizer(corpus, return_tensors='pt')
    token_ids = tokens['input_ids'].squeeze(0)
    print(f'Token size for {file_name} is {token_ids.shape}')
    torch.save(tokens['input_ids'], file_name)


def convert_to_integers(corpus, max_length, char_dict):
    '''
    converts letters in a sentence into a list of integers.
    42 would be the EOS token.
    '''

    embeddings = []
    masks = []

    eos_id = len(char_dict) + 1
    padding_id = len(char_dict) + 2

    for sentence in corpus:
        sentence_len = len(sentence)

        # initialize tensors for embedding and mask
        embedding = torch.full((max_length,), padding_id,
                               dtype=torch.long, device=device)
        mask = torch.full((max_length,), 0,
                          dtype=torch.float32, device=device)

        # bool for whether to use max_length or not
        # if False add EOS token after last character
        use_max_length = sentence_len >= max_length
        if not use_max_length:
            embedding[sentence_len] = eos_id
            mask[sentence_len + 1:] = float('-inf')

        loop_range = max_length if use_max_length else sentence_len
        for i in range(loop_range):
            char = sentence[i]
            if char in char_dict:
                embedding[i] = char_dict[char]
            else:
                embedding[i] = 0

        embeddings.append(embedding)
        masks.append(mask)

    return torch.stack(embeddings), torch.stack(masks)


def save_whole_letter_emb(corpus):
    '''
    Saves an entire corpus in letter embedding in a 1D tensor
    '''

    embeddings = []

    valid_chars = (" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.’'\"“”!?"
                   "0123456789—_[]:;\n")
    char_to_int = {char: idx for idx, char in enumerate(valid_chars)}
    print(char_to_int)

    for sentence in corpus:
        for char in sentence:
            if char in char_to_int:
                embeddings.append(char_to_int[char])
            else:
                embeddings.append(0)

    emb_tensor = torch.tensor(embeddings, dtype=torch.long)
    print(emb_tensor)
    print(emb_tensor.shape)

    torch.save(emb_tensor, 'emb_tensor.pt')


def train_tf(enc, dec, enc_mask, dec_mask, dec_ids, enc_num, dec_num,
             num_heads, vocab_size, batch_size, dropout=0.1,
             epochs=50, learning_rate=0.001, patience=5):
    emb_dim = 32
    model = Transformer(emb_dim, num_heads, vocab_size)
    # model = SimpleTransformer(vocab_size, emb_dim, num_heads, enc_num, dec_num, 2048, 512)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    check_for_nan_inf(enc, "enc_input")
    check_for_nan_inf(dec, "dec_input")
    check_for_nan_inf(enc_mask, "enc_mask")
    check_for_nan_inf(dec_mask, "dec_mask")

    dataset = CustomDataset2(enc, dec, enc_mask, dec_mask)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    patience_counter = 0

    print('Device:', torch.device(device))
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            enc_input = batch['enc'].to(device)
            enc_mask = batch['enc_mask'].to(device)
            dec_input = batch['dec'].to(device)
            dec_mask = batch['dec_mask'].to(device)

            enc_n_mask = torch.where(enc_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).to(device)
            dec_n_mask = torch.where(dec_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).to(device)

            outputs = model(enc_input, dec_input, enc_n_mask, dec_n_mask, enc_num, dec_num)

            ground_truth = dec_input if dec_ids is None else dec_ids
            batch_size, seq_length = dec_input.size()
            loss = criterion(outputs.view(batch_size * seq_length, vocab_size),
                             ground_truth.view(batch_size * seq_length))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                enc_input = batch['enc'].to(device)
                enc_mask = batch['enc_mask'].to(device)
                dec_input = batch['dec'].to(device)
                dec_mask = batch['dec_mask'].to(device)

                enc_n_mask = torch.where(enc_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).to(device)
                dec_n_mask = torch.where(dec_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).to(device)

                outputs = model(enc_input, dec_input, enc_n_mask, dec_n_mask, enc_num, dec_num)

                ground_truth = dec_input if dec_ids is None else dec_ids
                batch_size, seq_length = dec_input.size()
                loss = criterion(outputs.view(batch_size * seq_length, vocab_size),
                                 ground_truth.view(batch_size * seq_length))
                val_loss += loss.item()

        torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete")


def sliding_window(tensor, window_size=512, overlap=128):
    """
    Applies a sliding window with overlap on a 1D tensor.

    Parameters:
    tensor (numpy array): The input 1D tensor containing embeddings for each character.
    window_size (int): The size of each window.
    overlap (int): The number of overlapping characters.

    Returns:
    list of numpy arrays: A list containing windows of embeddings.
    """
    stride = window_size - overlap
    num_windows = (len(tensor) - window_size) // stride + 1
    windows = [tensor[i * stride: i * stride + window_size] for i in range(num_windows)]

    windows = torch.stack(windows).to(torch.long)
    print(windows)
    torch.save(windows, 'overlap_emb.pt')


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def load_enc_dec_files(file_list):
    '''
    Given a list of file names, loads the file name with _enc.pt and _dec.pt added to them
    Returns ids and masks from enc and dec files
    '''
    enc_id_list = []
    enc_mask_list = []
    dec_id_list = []
    dec_mask_list = []

    for file_name in file_list:
        enc_file_name = file_name + '_enc.pt'
        dec_file_name = file_name + '_dec.pt'

        enc_file = torch.load(enc_file_name)
        dec_file = torch.load(dec_file_name)

        enc_id_list.append(enc_file['ids'])
        enc_mask_list.append(enc_file['masks'])
        dec_id_list.append(dec_file['ids'])
        dec_mask_list.append(dec_file['masks'])

    enc_id_list = torch.cat(enc_id_list, dim=0).to('cpu')
    enc_mask_list = torch.cat(enc_mask_list, dim=0).to('cpu')
    dec_id_list = torch.cat(dec_id_list, dim=0).to('cpu')
    dec_mask_list = torch.cat(dec_mask_list, dim=0).to('cpu')

    print(f'Enc list shape is {enc_id_list.shape}')
    print(f'Dec list shape is {dec_id_list.shape}')

    return enc_id_list, enc_mask_list, dec_id_list, dec_mask_list

def load_enc_dec_files_letters(file_list):
    '''
    Same as above but for csv files.
    '''
    enc_id_list = []
    enc_mask_list = []
    dec_id_list = []
    dec_mask_list = []

    for file_name in file_list:
        data = torch.load(f'{file_name}_data.pt')

        enc_emb = data['enc_emb']
        enc_mask = data['enc_mask']
        dec_emb = data['dec_emb']
        dec_mask = data['dec_mask']

        enc_id_list.append(torch.stack(enc_emb, dim=0))
        enc_mask_list.append(torch.stack(enc_mask, dim=0))
        dec_id_list.append(torch.stack(dec_emb, dim=0))
        dec_mask_list.append(torch.stack(dec_mask, dim=0))

    emb_id_tensor = torch.cat(enc_id_list, dim=0)
    enc_mask_tensor = torch.cat(enc_mask_list, dim=0)
    dec_id_tensor = torch.cat(dec_id_list, dim=0)
    dec_mask_tensor = torch.cat(dec_mask_list, dim=0)

    print(emb_id_tensor.shape)
    print(enc_mask_tensor.shape)
    print(dec_id_tensor.shape)
    print(dec_mask_tensor.shape)

    return emb_id_tensor, enc_mask_tensor, dec_id_tensor, dec_mask_tensor

# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
vocab_size = 138
# print(vocab_size)

corpus_list = [
    'tom_sawyer',
    'don_quixote',
    'great_expectations',
    'jane_eyre',
    'karamazov_brothers',
    'les_miserables',
    'moby_dick',
    'war_and_peace'
]

emb1, mask1, emb2, mask2 = load_enc_dec_files_letters(corpus_list)

train_tf(emb1, emb2, mask1, mask2, None, 3, 3, 8, vocab_size, batch_size=128, epochs=100, learning_rate=0.001)