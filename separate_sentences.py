import spacy
import torch
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def read_text_file(file_path):
    with open(file_path + '.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def separate_sentences(txt_file, tokenizer, max_length):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.max_length = 4000000

    text = read_text_file(txt_file)

    # Segment the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # truncate sentences to max token length
    for i in range(len(sentences)):
        tokens = tokenizer.tokenize(sentences[i])
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            sentences[i] = tokenizer.convert_tokens_to_string(tokens)

    return sentences


def create_input_output_pairs(sentences, tokenizer, max_token_length, save_file_name):
    input_texts = []
    output_texts = []

    for i in range(len(sentences) - 1):
        current_input = sentences[i]
        current_output = sentences[i + 1]

        input_increment = 1
        output_increment = 2

        next_tokens_length = lambda x, y: len(tokenizer.tokenize(' '.join([x, sentences[i + y]])))
        next_input_tokens_length = next_tokens_length(current_input, input_increment)
        if i < len(sentences) - 2:
            next_output_tokens_length = next_tokens_length(current_output, output_increment)
        else:
            next_input_tokens_length = 512
            next_output_tokens_length = 512

        while (next_input_tokens_length <= max_token_length
               and i + input_increment <= len(sentences) - 1):
            current_input = ' '.join([current_input, sentences[i + input_increment]])
            input_increment += 1
            try:
                next_input_tokens_length = next_tokens_length(current_input, input_increment)
            except:
                break

        while (next_output_tokens_length <= max_token_length
               and i + output_increment <= len(sentences) - 1):
            current_output = ' '.join([current_output, sentences[i + output_increment]])
            output_increment += 1
            try:
                next_output_tokens_length = next_tokens_length(current_output, output_increment)
            except:
                break

        input_texts.append(current_input)
        output_texts.append(current_output)

    # generate ids and masks for both inputs and outputs
    input_tokenizer = tokenizer(input_texts, padding=True, truncation=True,
                                max_length=max_token_length, return_tensors='pt')
    input_ids = input_tokenizer['input_ids']
    input_masks = input_tokenizer['attention_mask']
    input_dict = {
        'ids': input_ids,
        'masks': input_masks
    }
    torch.save(input_dict, save_file_name + '_enc.pt')

    output_tokenizer = tokenizer(output_texts, padding=True, truncation=True,
                                 max_length=max_token_length, return_tensors='pt')
    output_ids = output_tokenizer['input_ids']
    output_masks = output_tokenizer['attention_mask']
    output_dict = {
        'ids': output_ids,
        'masks': output_masks
    }
    torch.save(output_dict, save_file_name + '_dec.pt')

    print(input_dict)
    print(output_dict)

    return


def inspect_token_length(sentences, tokenizer):
    token_lengths = [len(tokenizer.tokenize(sentence)) for sentence in sentences]
    print(f'Max token lengths for this corpus is {max(token_lengths)}')

    # Plot the distribution of token lengths
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=range(1, max(token_lengths) + 2), edgecolor='black')
    plt.title('Distribution of Token Lengths in Sentences')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.show()

    return


def separate_sentences_letter_length(txt_file, max_length):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.max_length = 4000000

    text = read_text_file(txt_file)

    # Segment the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    sentences_count = len(sentences)
    print(f'Total sentence length is {sentences_count}')

    output = []

    for i in range(sentences_count):
        current_sentence = sentences[i]
        increment = 1
        while len(current_sentence) <= 512:
            try:
                appended_sentence = current_sentence + ' ' + sentences[i + increment]
            except IndexError:
                print('End of index reached, breaking')
                break
            if len(appended_sentence) <= 512:
                current_sentence = appended_sentence
            else:
                break

        if len(current_sentence) > 512:
            current_sentence = current_sentence[:512]

        output.append(current_sentence)

    sentence_lengths = [len(sentence) for sentence in output]

    # Plot the distribution of token lengths
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=range(1, max(sentence_lengths) + 2), edgecolor='black')
    plt.title('Distribution of Token Lengths in Sentences')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.show()

    return output


def generate_sentence_pairs_letters(sentences, seq_length, char_dict, save_file_name):

    embeddings = []
    masks = []
    enc_emb = []
    enc_mask = []
    dec_emb = []
    dec_mask = []

    # generate embedding and masks for the sentences
    # 0 is for padding token and 1 is for EOS
    for sentence in sentences:
        embedding = torch.zeros((seq_length,), dtype=torch.long)
        mask = torch.ones((seq_length,), dtype=torch.long)
        sentence_length = len(sentence)

        if sentence_length <= seq_length - 1:
            embedding[sentence_length] = 1
            try:
                mask[sentence_length + 1:] = 0
            except IndexError:
                pass

        for i in range(sentence_length):
            try:
                if sentence[i] in char_dict:
                    embedding[i] = char_dict[sentence[i]]
                else:
                    embedding[i] = char_dict[' ']
            except IndexError:
                print(f'Character attempted to parse is {sentence[i]}')
                return

        embeddings.append(embedding)
        masks.append(mask)

    # generate pairs
    for i in range(len(embeddings) - 1):
        enc_emb.append(embeddings[i])
        enc_mask.append(masks[i])
        dec_emb.append(embeddings[i + 1])
        dec_mask.append(masks[i + 1])

    data = {'enc_emb': enc_emb, 'enc_mask': enc_mask,
            'dec_emb': dec_emb, 'dec_mask': dec_mask}

    torch.save(data, f'{save_file_name}_data.pt')

    return




def preprocess_texts(corpus_list, max_length):
    '''
    Takes in a list of file names and preprocesses data for each of them.
    '''

    for file_name in corpus_list:
        sentences = separate_sentences(file_name, tokenizer, max_length)
        print(f'Length of sentences is {len(sentences)}')

        inspect_token_length(sentences, tokenizer)

        create_input_output_pairs(sentences, tokenizer, 512, file_name)


def preprocess_texts_letters(corpus_list, max_length):
    '''
    Takes in a list of file names and preprocesses data for each of them.
    '''

    corpus_sentences_list = []
    total_sentences_length = 0
    unique_chars = set()

    for file_name in corpus_list:
        sentences = separate_sentences_letter_length(file_name, max_length)
        sentences_length = len(sentences)
        print(f'Length of sentences is {sentences_length} for {file_name}.')
        total_sentences_length += sentences_length
        corpus_sentences_list.append(sentences)

    print(f'In total, we have {total_sentences_length} sentences.')

    for corpus in corpus_sentences_list:
        chars = set("".join(corpus))
        unique_chars.update(chars)

    sorted_chars = sorted(unique_chars)
    char_dict = {char: index + 2 for index, char in enumerate(sorted_chars)}
    print(char_dict)
    torch.save(char_dict, 'char_dict.pt')

    for corpus, file_name in zip(corpus_sentences_list, corpus_list):
        generate_sentence_pairs_letters(corpus, max_length, char_dict, file_name)





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


preprocess_texts_letters(corpus_list, 512)



# preprocess_texts(corpus_list, 512)
