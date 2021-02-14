import json
import re

import itertools
import numpy as np
import matplotlib.pyplot as plt

from first_assignment.LP_toolkits import normalizer, tokenizer
from first_assignment.constants import given_data_root_path, document_root_path, frequency_limitation, law_limit


def create_vocabulary_json_file(v):
    with open(document_root_path + '\\vocabulary_file.json', 'w', encoding='utf-8') as vocabulary_file:
        json.dump(v, vocabulary_file, ensure_ascii=False)


# #     1.3:  reporting total_tokens_# and total_unique_tokens_#
def report_token_number():
    with open(document_root_path + '\\vocabulary_file.json', 'r', encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        print('total token number:              ', sum(vocabulary.values()))
        print('total unique token number:       ', len(vocabulary))


# #     1.4:    save 10000 words with highest frequency in most_frequent.txt file using sorted vocabulary json file
def most_frequent_words():
    with open(document_root_path + '\\vocabulary_file.json', 'r', encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        most_frequent_words_dictionary = dict(itertools.islice(vocabulary.items(), frequency_limitation))
        with open('most_frequent.txt', 'w', encoding='utf8') as most_frequent_file:
            for word in most_frequent_words_dictionary:
                most_frequent_file.write(word + '\n')


# #    1.5: print 10000 words with highest frequency percentage among all tokens (with 2 decimal numbers)
def most_frequent_words_percentage_among_all():
    with open(document_root_path + 'vocabulary_file.json', encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        most_frequent_words_dictionary = dict(itertools.islice(vocabulary.items(), frequency_limitation))

        frequent_token_counter = 0
        token_counter = 0
        # #     count all the tokens in training data
        for word in vocabulary.keys():
            token_counter += vocabulary[word]
        # #     count 10000 words with highest frequency appearance in training data
        for word in most_frequent_words_dictionary.keys():
            frequent_token_counter += most_frequent_words_dictionary[word]

        # print(frequent_token_counter)
        # print(token_counter)
        print('most frequent words percentage among all the tokens:     {:.2f}'.format(
            (frequent_token_counter / token_counter) * 100))


# #     1.6 ,1.7 :  replace least frequent tokens with 'UKN' (1.6)
# #              and save eah sentence (recognized by dot and question mark) and save in text file in separate lines

def reformat_least_frequent_words(train_doc_array):
    most_frequent_word = []
    with open('most_frequent.txt', 'r', encoding='utf8') as most_frequent_file:
        most_frequent_word_read = most_frequent_file.read()
        for line in most_frequent_word_read.split():
            most_frequent_word.append(line.split()[0])

    with open(document_root_path + 'train_sentences.txt', 'w', encoding='utf8') as train_sentences_file:

        for news in train_doc_array:
            # #     first find least frequent words for each news to replace them with 'UKN'
            news_word_array = tokenizer(news)
            least_frequent = []
            for word in news_word_array:
                if word not in most_frequent_word:
                    least_frequent.append(word)
            # #     split news content by space then if each term is in the least frequent set, then replace it with 'UKN"
            # #     didn't used replace function as it can raise some issue  by replacing substrings of most frequent words
            news_array = news.split()
            for i in range(len(news_array)):
                if news_array[i] in least_frequent:
                    news_array[i] = 'UKN'

            # save the fully normalized sentenses to train_sentences.txt file
            sentences = re.split('[.؟?]', ' '.join(news_array))
            for sentence in sentences:
                if not sentence == ' ' and len(sentence) > 0:
                    train_sentences_file.write(sentence + '\n')


# #     1.1, 1.2, 1.4, 1.6, and 1.7 of the assignment
def read_training_data(raw_train_corpus_dir):
    normalized_train_doc = []
    with open(raw_train_corpus_dir, 'r', encoding='utf8') as train_json_file:
        vocabulary = {}
        train_list = json.load(train_json_file)

        for each in train_list:
            news = normalizer(each)

            # #     save normalized news to use them later
            normalized_train_doc.append(news)

            news_term_array = tokenizer(news)

            # #     generate vocabulary based on the training data while counting each word appearance
            # #     structured as a dictionary

            for word in news_term_array:
                if word in vocabulary.keys():
                    vocabulary[word] += 1
                else:
                    vocabulary.update({word: 1})

    sorted_vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)}

    create_vocabulary_json_file(sorted_vocabulary)
    most_frequent_words()
    reformat_least_frequent_words(normalized_train_doc)


# #     1.8:    power law
def power_law():
    x_frequency = []
    y_rank = []

    with open(document_root_path + '\\vocabulary_file.json', 'r', encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        most_frequent_words_dictionary = dict(itertools.islice(vocabulary.items(), law_limit))
        for num in most_frequent_words_dictionary.values():
            if num != 0:
                x_frequency.append(np.log10(num))
            else:
                x_frequency.append(0)
        for num in range(1, law_limit + 1):
            if num != 0:
                y_rank.append(np.log10(num))
            else:
                y_rank.append(0)

    plt.plot(x_frequency, y_rank, label="training data")

    plt.xlabel('log10 Term Frequency')

    plt.ylabel('log10 Term Rank')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    print('train_data_prep')
    # includes all the process to create a normalized text and save it in a text file
    # which would be parts 1.1, 1.2, 1.4, 1.6, 1.7 of the assignment
    read_training_data(given_data_root_path + '\\train.json')
    # part 1.3 of the assignment which report all tokens # and the unique tokens #
    report_token_number()
    # part 1.5 of the assignment which show most frequent words tokens among all the tokens
    most_frequent_words_percentage_among_all()
    # part 1.8 of the assignment which plot the asked graph in order to check the hypothesis
    power_law()
