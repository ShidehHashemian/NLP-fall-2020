import json
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from constant import given_doc_root_path, document_root_path, frequency_limitation
from LP_toolkits import normalizer
from preprocessing import reformat_least_frequent_words


# # 3.6:    get most_frequent.txt file add (as all words are in or are 'UKN')
# #         and construct word2index, index2word,char2index,index2char using them
def word_char_index_construction(most_frequent_file_add):
    """
    :param most_frequent_file_add: an address string of most_frequent.txt text file
    :return:

    save four dictionary as a pickle file named word2index.pickle, index2word.pickle,
     char2index.pickle and index2char.pickle
    """
    word2index = dict()
    index2word = dict()
    char2index = dict()
    index2char = dict()
    words = []
    chars = []
    word_counter = 0
    char_counter = 0
    with open(most_frequent_file_add, 'r', encoding='utf8') as most_frequent_file:
        most_frequent_word_read = most_frequent_file.read()

        for line in most_frequent_word_read.split():
            words.append(line.split()[0])
            for ch in words[-1]:
                if ch not in chars:
                    chars.append(ch)

        words.append('UKN')
        for ch in words[-1]:
            if ch not in chars:
                chars.append(ch)
        chars.append(' ')
    # for the added PAD in part 3.7
    words.append('PAD')
    chars.append('PAD')

    words.sort()
    chars.sort()

    for word in words:
        word2index.update({word: word_counter})
        index2word.update({word_counter: word})
        word_counter += 1
    for char in chars:
        char2index.update({char: char_counter})
        index2char.update({char_counter: char})
        char_counter += 1

    with open(document_root_path + 'word2index.pickle', 'wb') as word2index_file:
        pickle.dump(word2index, word2index_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(document_root_path + 'index2word.pickle', 'wb') as index2word_file:
        pickle.dump(index2word, index2word_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(document_root_path + 'char2index.pickle', 'wb') as char2index_file:
        pickle.dump(char2index, char2index_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(document_root_path + 'index2char.pickle', 'wb') as index2char_file:
        pickle.dump(index2char, index2char_file, protocol=pickle.HIGHEST_PROTOCOL)


# # 3.6:    get word2index dictionary and news_arr, and return news__indexed_arr base in the given indexing
def word_indexing(word2index, train_news_arr):
    """

    :param word2index: a dictionary which it's keys are words and values are words' index
    :param train_news_arr: an array of news (which they length has been homogenized)
    :return: an array of indexed news based on their words
    """
    train_news_arr_indexed = list()
    for news in train_news_arr:
        news_index_arr = list()
        for word in news.split():
            news_index_arr.append(word2index[word])
        train_news_arr_indexed.append(news_index_arr)
    return train_news_arr_indexed


# # 3.6:    get char2index dictionary and news_arr, and return news__indexed_arr base in the given indexing
def char_indexing(char2index, train_news_arr):
    """
    :param char2index: a dictionary which it's keys are chars and values are chars' index
    :param train_news_arr: an array of news (which they length has been homogenized)
    :return: an array of indexed news based on their chars
    """
    train_news_arr_indexed = list()
    for news in train_news_arr:
        news_index_arr = list()
        counter = 0
        for char in news:
            if char != 'P':
                news_index_arr.append(char2index[char])
                counter += 1
            else:
                for i in range(int((len(news) - counter) / 3)):
                    news_index_arr.append(char2index['PAD'])
                break
        train_news_arr_indexed.append(news_index_arr)
    return train_news_arr_indexed


# # 3.7:    handle docs length (word base)
def handle_docs_length_word_level(news_arr):
    """
    :param news_arr: an array of news (with different lengths)
    :return: an array of news which have same length (here means same word count)
    """
    avg_len = 0
    # fist calculate avg of docs words
    for news in news_arr:
        avg_len += len(news.split())
    avg_len /= len(news_arr)
    avg_len = int(avg_len)

    # list of shortened news
    shortened_news_arr = list()
    news_id = set()
    counter = 0
    # now only add those that have smaller or equal length to the avg,
    # and add 'PAD' word to those that have shorten length
    for news in news_arr:
        if len(news.split()) <= avg_len:
            news_id.add(counter)
            # print(news)
            for i in range(avg_len - len(news.split())):
                news += ' PAD'
            shortened_news_arr.append(news)
        counter += 1

    return shortened_news_arr, news_id  # Return list of shortened news and
    # list of the news that still we're using, remove category for the removed one


# # 3.7:    handle docs length (char base)
def handle_docs_length_char_level(news_arr):
    """

    :param news_arr: an array of news (with different lengths)
    :return: an array of news with same length (here means same char counts(space has been included as a char))
    """
    avg_len = 0
    # fist calculate avg of docs words
    for news in news_arr:
        avg_len += len(news.split())
    avg_len /= len(news_arr)
    avg_len = int(avg_len)

    max_length = 0
    # list of shortened news
    shortened_news_arr = list()
    news_id = []
    counter = 0
    # now find the enhance with the max char between those that have equal or smaller words than avg
    # and also add them to shortened_news_arr
    for news in news_arr:
        if len(news.split()) <= avg_len:
            news_id.append(counter)
            counter += 1
            shortened_news_arr.append(news)
            if len(news) > max_length:
                max_length = len(news)
    # now add 'PAD' char to those that have smaller count of char than the max_length.
    for i in range(len(shortened_news_arr)):
        if len(shortened_news_arr[i]) <= max_length:

            for num in range(max_length - len(shortened_news_arr[i])):
                shortened_news_arr[i] += 'PAD'

    return shortened_news_arr, news_id  # Return list of shortened news.
    # list of the news that still we're using, remove category for the removed one


# # 4.1:
def clean(raw_news_arr):
    """

    :param raw_news_arr: an array of news
    :return: an array of news which have been normalized and replaced less_frequent words with 'UKN'
            and numbers with 'N'
    """
    news_arr = []
    for news in raw_news_arr:
        news_arr.append(normalizer(news))
        reformat_least_frequent_words(news_arr, document_root_path + 'most_frequent.txt')
    # As length_uniformed newses differ for char_level and word_level
    # we consider applying it in tokenizer as by then we know which type we're using
    return news_arr


# # 4.2:
def tokenize(news_cleaned_arr, word2index, char2index, level):
    """
    :param news_cleaned_arr: an array of news which have been cleaned (those item mentioned in clean function)
    :param word2index: a dictionary which it's keys are words and values are words' index
    :param char2index: a dictionary which it's keys are chars and values are chars' index
    :param level: an int which can be 0: word_level,1: char_level or 2: word_&_char_level
    :return: an array of indexed news base on the level (if level = 2 it returns two array)
    """
    if level == 0:
        news_word, n_id = handle_docs_length_word_level(news_cleaned_arr)

        # remove category of removed doc and add remained one to the  train_y_vec
        train_y_vec = []
        with open(document_root_path + 'train_category.pickle', 'rb') as train_category_vector_file:
            train_category = pickle.load(train_category_vector_file, encoding='utf8')
            for i in range(len(train_category)):
                if i in n_id:
                    train_y_vec.append(train_category[i])

        # in 'train_category.pickle' file to use is as Y vector for training
        with open(document_root_path + 'train_y_vec.pickle', 'wb') as train_y_file:
            pickle.dump(train_y_vec, train_y_file, protocol=pickle.HIGHEST_PROTOCOL)

        return word_indexing(word2index, news_word)  # Return word_indexing array of docs
    elif level == 1:
        news_char, n_id = handle_docs_length_char_level(news_cleaned_arr)

        # remove category of removed doc and add remained one to the  train_y_vec
        train_y_vec = []
        with open(document_root_path + 'train_category.pickle', 'rb') as train_category_vector_file:
            train_category = pickle.load(train_category_vector_file, encoding='utf8')
            for i in range(len(train_category)):
                if i in n_id:
                    train_y_vec.append(train_category[i])

        # in 'train_category.pickle' file to use is as Y vector for training
        with open(document_root_path + 'train_y_vec.pickle', 'wb') as train_y_file:
            pickle.dump(train_y_vec, train_y_file, protocol=pickle.HIGHEST_PROTOCOL)

        return char_indexing(char2index, news_char)  # Return char_indexing array of docs
    elif level == 2:
        news_char, n_id = handle_docs_length_char_level(news_cleaned_arr)
        news_word, n_id = handle_docs_length_word_level(news_cleaned_arr)
        # remove category of removed doc and add remained one to the  train_y_vec
        train_y_vec = []
        with open(document_root_path + 'train_category.pickle', 'rb') as train_category_vector_file:
            train_category = pickle.load(train_category_vector_file, encoding='utf8')
            for i in range(len(train_category)):
                if i in n_id:
                    train_y_vec.append(train_category[i])

        # in 'train_category.pickle' file to use is as Y vector for training
        with open(document_root_path + 'train_y_vec.pickle', 'wb') as train_y_file:
            pickle.dump(train_y_vec, train_y_file, protocol=pickle.HIGHEST_PROTOCOL)

        return word_indexing(word2index, news_word), char_indexing(char2index, news_char)
        # Return word_indexing,char_indexing array of docs


# # 3.8:    construct count_vec for docs to use it latter to construct tf_idf vectors
def count_vector_constructor(doc_indexed_arr, features_index):
    """
    :param doc_indexed_arr: an array of indexed news
    :param features_index: an array of features index (it could be keys of index2word or word2index base on the level)
    :return: a 2_dimensions array which array[i][j] shows number of appearance of feature j in document i
    """
    doc_count_vec = list()
    for news_arr in doc_indexed_arr:
        count_arr = list()
        for feature in features_index:
            count_arr.append(news_arr.count(feature))
        doc_count_vec.append(count_arr)
    return doc_count_vec  # Return doc_count_vec


# # 3.9:    split docs into 10 chunk to use them separately to not face memory shortened problem
def sub_10_chunk(doc_indexed_arr):
    """
    :param doc_indexed_arr: an array of indexed news
    :return: an array which has each chunk as an object
    """
    doc_chunks = list()

    for i in range(10):
        doc_chunks.append(list())

    for i in range(len(doc_indexed_arr)):
        doc_chunks[i % 10].append(doc_indexed_arr[i])

    return doc_chunks


# # 4.2:
def vectorize(word_indexed_doc_arr, char_indexed_doc_arr, index2word, index2char, level):
    """

    :param word_indexed_doc_arr: an array of news indexed word_level (not None when level =0|2)
    :param char_indexed_doc_arr: an array of news indexed char_level (not None when level =1|2)
    :param index2word: a dictionary which it's keys are indexes and values are indexes' word
    :param index2char: a dictionary which it's keys are indexes and values are indexes' char
    :param level: an int which can be 0: word_level,1: char_level or 2: word_&_char_level
    :return:

    save 10 pickle file that each of them is a chunk of news' vectors arr
    each can be seen as train_x_word_level_{#}.pickle or train_x_char_level_{#}.pickle files based on the level
    """
    if level == 0:
        chunks = sub_10_chunk(word_indexed_doc_arr)
        counter = 0
        # extract features indexes to search them in the doc_indexed_arr and return the doc_count_vec
        # which is a vec of each features counts in each doc
        features_index = list(index2word.keys())
        tfidf_transformer = TfidfTransformer()
        for indexed_news in chunks:
            # # 3.8: first construct count_vec for each chunk and then tf_idf vectorization
            x_train_tfidf = tfidf_transformer.fit_transform(count_vector_constructor(indexed_news, features_index))
            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(counter), 'wb') as train_x_file:
                pickle.dump(x_train_tfidf, train_x_file, protocol=pickle.HIGHEST_PROTOCOL)
            print('vectorized word_model chunk {}'.format(counter))
            counter += 1

    elif level == 1:

        chunks = sub_10_chunk(char_indexed_doc_arr)
        counter = 0
        # extract features indexes to search them in the doc_indexed_arr and return the doc_count_vec
        # which is a vec of each features counts in each doc
        features_index = list(index2char.keys())
        tfidf_transformer = TfidfTransformer()
        for indexed_news in chunks:
            x_train_tfidf = tfidf_transformer.fit_transform(count_vector_constructor(indexed_news, features_index))
            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(counter), 'wb') as train_x_file:
                pickle.dump(x_train_tfidf, train_x_file, protocol=pickle.HIGHEST_PROTOCOL)
            print('vectorized char_model chunk {}'.format(counter))

            counter += 1

    elif level == 2:
        chunks = sub_10_chunk(word_indexed_doc_arr)
        counter = 0
        # extract features indexes to search them in the doc_indexed_arr and return the doc_count_vec
        # which is a vec of each features counts in each doc
        features_index = list(index2word.keys())
        tfidf_transformer = TfidfTransformer()
        for indexed_news in chunks:
            x_train_tfidf = tfidf_transformer.fit_transform(count_vector_constructor(indexed_news, features_index))
            with open(document_root_path + 'train_x_word_level_{}_.pickle'.format(counter), 'wb') as train_x_file:
                pickle.dump(x_train_tfidf, train_x_file, protocol=pickle.HIGHEST_PROTOCOL)
            print('vectorized word_model chunk {}'.format(counter))
            counter += 1

        chunks = sub_10_chunk(char_indexed_doc_arr)
        counter = 0
        # extract features indexes to search them in the doc_indexed_arr and return the doc_count_vec
        # which is a vec of each features counts in each doc
        features_index = list(index2char.keys())
        for indexed_news in chunks:
            x_train_tfidf = tfidf_transformer.fit_transform(count_vector_constructor(indexed_news, features_index))
            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(counter), 'wb') as train_x_file:
                pickle.dump(x_train_tfidf, train_x_file, protocol=pickle.HIGHEST_PROTOCOL)

            print('vectorized char_model chunk {}'.format(counter))

            counter += 1


# # 4.3:
def defining_model():
    """

    :return: an SGDClassifier object that classy as an SVM classifier using gradient descent
    """
    # using SGD instead of SVC as it can be trained with chunks and itwe don't have to get all the data at the same time
    clf = SGDClassifier(loss="hinge", penalty="l2")
    return clf


# # 4.4:
def train(level):
    """
    :param level: an int which can be 0: word_level or 1: char_level
    :return:

    train based on the saved chunks and categories by applying cross-validation
    (train five model, each time use just 8 chunks and validate the model using 2 left ones)
    then save the model object on a pickle file

    use 'partial_fit' instead of 'fit' function as it all the data may not be able to be stored in the memory
    and by using 'partial_fit' we can pass the training set in chunks (we should also call this function for each data
    chunk several times as it doesn't have any conditional iterative loop on it, here we considered 300 times  )

    """
    if level == 0:
        train_y_chunk = list()
        # classes to pass to training func.
        cat = np.array([])
        #  save the same chunks for the categories
        for i in range(10):
            train_y_chunk.append(list())
        with open(document_root_path + 'train_y_vec.pickle', 'rb') as train_y_file:
            train_y = pickle.load(train_y_file, encoding='utf8')
            cat = np.array(train_y)
            for i in range(len(train_y)):
                train_y_chunk[i % 10].append(train_y[i])

        clf = defining_model()
        max_accuracy = 0.
        cross = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                 [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                 [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                 [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]]

        for turn in cross:
            print('********************   {}   ********************'.format(turn))
            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[0]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[0]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[0]))
            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[1]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[1]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[1]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[2]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[2]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[2]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[3]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[3]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[3]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[4]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[4]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[4]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[5]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[5]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[5]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[6]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[6]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[6]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[7]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[7]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[7]))

            with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[8]), 'rb') as train_8:
                with open(document_root_path + 'train_x_word_level_{}.pickle'.format(turn[9]), 'rb') as train_9:
                    a = pickle.load(train_8)

                    b = pickle.load(train_9)

                    c1 = clf.score(a, train_y_chunk[turn[8]])
                    c2 = clf.score(b, train_y_chunk[turn[9]])
                    c = (c1 + c2) / 2
                    if c > max_accuracy:
                        max_accuracy = c
                        print(max_accuracy)
                        with open(document_root_path + 'SVM_model_word_base.pickle', 'wb') as model_file:
                            pickle.dump(clf, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    elif level == 1:
        train_y_chunk = list()
        # classes to pass to training func.
        cat = np.array([])
        #  save the same chunks for the categories
        for i in range(10):
            train_y_chunk.append(list())
        with open(document_root_path + 'train_y_vec.pickle', 'rb') as train_y_file:
            train_y = pickle.load(train_y_file, encoding='utf8')
            cat = np.array(train_y)
            for i in range(len(train_y)):
                train_y_chunk[i % 10].append(train_y[i])

        clf = defining_model()
        max_accuracy = 0.
        cross = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                 [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                 [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                 [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]]

        for turn in cross:
            print('********************   {}   ********************'.format(turn))
            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[0]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[0]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[0]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[1]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[1]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[1]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[2]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[2]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[2]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[3]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[3]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[3]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[4]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[4]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[4]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[5]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[5]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[5]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[6]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[6]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[6]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[7]), 'rb') as r:
                tf_idf = pickle.load(r)
                y = np.array(train_y_chunk[turn[7]])

                for i in range(300):
                    clf.partial_fit(tf_idf, y, classes=np.unique(cat))
                print('trained chunk {}'.format(turn[7]))

            with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[8]), 'rb') as train_8:
                with open(document_root_path + 'train_x_char_level_{}.pickle'.format(turn[9]), 'rb') as train_9:
                    a = pickle.load(train_8)

                    b = pickle.load(train_9)

                    c1 = clf.score(a, train_y_chunk[turn[8]])
                    c2 = clf.score(b, train_y_chunk[turn[9]])
                    c = (c1 + c2) / 2
                    if c > max_accuracy:
                        max_accuracy = c
                        print(max_accuracy)
                        with open(document_root_path + 'SVM_model_char_base.pickle', 'wb') as model_file:
                            pickle.dump(clf, model_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('Classifier')

    # construction word2index,... dictionary
    word_char_index_construction(document_root_path + 'most_frequent.txt')
    train_news = list()

    word2index = dict()
    index2word = dict()
    char2index = dict()
    index2char = dict()
    # load callings requirements in memory
    with open(document_root_path + 'train_news.pickle', 'rb') as train_news_file:
        train_news = pickle.load(train_news_file, encoding='utf8')

    with open(document_root_path + 'word2index.pickle', 'rb') as word2index_file:
        word2index = pickle.load(word2index_file, encoding='utf8')

    with open(document_root_path + 'char2index.pickle', 'rb') as char2index_file:
        char2index = pickle.load(char2index_file, encoding='utf8')

    with open(document_root_path + 'index2word.pickle', 'rb') as index2word_file:
        index2word = pickle.load(index2word_file, encoding='utf8')

    with open(document_root_path + 'index2char.pickle', 'rb') as index2char_file:
        index2char = pickle.load(index2char_file, encoding='utf8')

    # calling tokenize function word_level
    token_arr_word_level = tokenize(train_news, word2index, char2index, 0)

    # calling vectorize function word_level
    vectorize(token_arr_word_level, [], index2word, index2char, 0)

    # train model word_level
    train(0)
