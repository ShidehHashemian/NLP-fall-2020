import pandas as pd
import json
import pickle
import itertools

from constant import given_doc_root_path, document_root_path, frequency_limitation
from LP_toolkits import normalizer


def read_csv_train_data(doc_add):
    """
    :param doc_add: a string includes the training set address
    :return:

    normalize news text and extract 10000 most frequent words, then replace others with 'UKN' then save the news arr to
    'train_news.pickle' and their categories arr to 'train_category.pickle'

    """
    csv_data = pd.read_csv(doc_add, delimiter='\t')

    doc_collection = []
    category = []
    vocabulary = {}
    for index in range(csv_data.shape[0]):
        text = normalizer(str(csv_data['text'][index]))
        doc_collection.append(text)
        category.append(str(csv_data['category'][index]))

        news_term_arr = text.split()

        # generate vocabulary based on the training data while counting each word appearance
        # structured as a vocabulary

        for word in news_term_arr:
            if word in vocabulary.keys():
                vocabulary[word] += 1
            else:
                vocabulary.update({word: 1})

    sorted_vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)}

    with open(document_root_path + 'vocabulary.json', 'w', encoding='utf8') as vocabulary_file:
        json.dump(sorted_vocabulary, vocabulary_file, ensure_ascii=False)

    most_frequent_words(document_root_path + 'vocabulary.json')

    doc_collection = reformat_least_frequent_words(doc_collection, document_root_path + 'most_frequent.txt')

    with open(document_root_path + 'train_news.pickle', 'wb') as train_file:
        pickle.dump(doc_collection, train_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(document_root_path + 'train_category.pickle', 'wb') as train_collection_file:
        pickle.dump(category, train_collection_file, protocol=pickle.HIGHEST_PROTOCOL)


# # 3.3:   return chars_number, words_number
def char_word_number(vocabulary_json_add):
    chars_num = 0
    words_num = 0

    with open(vocabulary_json_add, 'r', encoding='utf8') as vocabulary_json_file:
        vocabulary = json.load(vocabulary_json_file)
        chars = set()
        words_num = len(vocabulary.keys())

        for word in vocabulary.keys():

            for ch in word:
                chars.update(ch)
        chars_num = len(chars)

    # return chars_num, words_num
    print('char #        {}'.format(chars_num))
    print('word #        {}'.format(words_num))


# # 3.4:    find 10000 most frequent word and save them in most_frequent.txt
def most_frequent_words(vocabulary_json_add):
    with open(vocabulary_json_add, 'r', encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        most_frequent_words_vocabulary = dict(itertools.islice(vocabulary.items(), frequency_limitation))

        with open(document_root_path + 'most_frequent.txt', 'w', encoding='utf8') as most_frequent_file:
            for word in most_frequent_words_vocabulary:
                most_frequent_file.write(word + '\n')


# # 3.4:    print 10000 words with highest frequency percentage among all tokens (with 2 decimal numbers)
def most_frequent_words_percentage_among_all(vocabulary_json_add):
    with open(vocabulary_json_add, encoding='utf8') as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
        most_frequent_words_dictionary = dict(itertools.islice(vocabulary.items(), frequency_limitation))

        frequent_token_counter = 0
        token_counter = 0
        #      count all the tokens in training data
        for word in vocabulary.keys():
            token_counter += vocabulary[word]
        #      count 10000 words with highest frequency appearance in training data
        for word in most_frequent_words_dictionary.keys():
            frequent_token_counter += most_frequent_words_dictionary[word]

        # print(frequent_token_counter)
        # print(token_counter)
        print('most frequent words percentage among all the tokens:     {:.2f}'.format(
            (frequent_token_counter / token_counter) * 100))


# # 3.5:     get docs text array and most_frequent.txt add and replace not_frequent words with UKN in news texts
def reformat_least_frequent_words(train_doc_arr, most_frequent_file_add):
    most_frequent_word = set()
    with open(most_frequent_file_add, 'r', encoding='utf8') as most_frequent_file:
        most_frequent_word_read = most_frequent_file.read()
        for line in most_frequent_word_read.split():
            most_frequent_word.add(line.split()[0])
    train_reformed_news_arr = []
    for news in train_doc_arr:
        #   replace all words which are not in the most_frequent.txt with 'UKN'
        news_word_arr = news.split()
        for i in range(len(news_word_arr)):
            if news_word_arr[i] not in most_frequent_word:
                news_word_arr[i] = 'UKN'
        #  then replace old news with the processed one
        train_reformed_news_arr.append(' '.join(news_word_arr))
    return train_reformed_news_arr


def char_word_txt_constructor(most_frequent_file_add):
    most_frequent_word = set()
    with open(most_frequent_file_add, 'r', encoding='utf8') as most_frequent_file:
        most_frequent_word_read = most_frequent_file.read()
        for line in most_frequent_word_read.split():
            most_frequent_word.add(line.split()[0])

    most_frequent_word.add('UKN')
    chars = set()
    for word in most_frequent_word:
        for char in word:
            chars.update(char)
    chars.update(' ')
    with open(document_root_path + 'words.txt', 'w', encoding='utf8') as words_file:
        for word in most_frequent_word:
            words_file.write(word + '\n')
    with open(document_root_path + 'chars.txt','w', encoding='utf8') as chars_file:
        for ch in chars:
            chars_file.write(ch + '\n')


if __name__ == '__main__':
    print('preprocessing')

    # reading train.csv data an normalize them
    read_csv_train_data(given_doc_root_path + 'train.csv')
    # passing vocabulary add and save 1000 most frequent word in most_frequent.txt
    most_frequent_words(document_root_path + 'vocabulary.json')
    # print word_unique_# and char_unique_#
    char_word_number(document_root_path + 'vocabulary.json')
    # print 1000 most frequent words percentage among all
    most_frequent_words_percentage_among_all(document_root_path + 'vocabulary.json')
    # construction words.txt and chars.txt based on the remained words after cleaning
    char_word_txt_constructor(document_root_path + 'most_frequent.txt')
