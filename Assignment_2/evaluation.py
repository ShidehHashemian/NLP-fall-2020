import pandas as pd
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, \
    roc_curve, auc

import matplotlib.pyplot as plt

from constant import given_doc_root_path, document_root_path
from LP_toolkits import normalizer
from preprocessing import reformat_least_frequent_words
from classifier import handle_docs_length_word_level, handle_docs_length_char_level, word_indexing, char_indexing, \
    count_vector_constructor


def test_doc_prep(test_doc_add, index2word, word2index, index2char, char2index, level):
    """

    :param test_doc_add: address of test.csv file
    :param index2word: index2word dictionary
    :param word2index: word2index dictionary
    :param index2char: index2char dictionary
    :param char2index: char2index dictionary
    :param level: training level, an int between 0: word_level and 1: char_level
    :return:

    do the same text normalization and indexing preprocess and vectorize news' text in test.csv
    and save them in 'test_news_word.pickle' or 'test_news_char.pickle' based on the given level value
    and also save the given classes for test set in 'test_category.pickle'
    """
    csv_data = pd.read_csv(test_doc_add, delimiter='\t')

    doc_collection = list()
    category = list()
    for index in range(csv_data.shape[0]):
        text = normalizer(str(csv_data['text'][index]))
        doc_collection.append(text)
        category.append(str(csv_data['category'][index]))

    doc_collection = reformat_least_frequent_words(doc_collection, document_root_path + 'most_frequent.txt')

    tfidf_transformer = TfidfTransformer()
    if level == 0:  # Word_level
        # handle sentences length
        doc_collection, n_id = handle_docs_length_word_level(doc_collection)
        # remove category of removed doc and add remained one to the  train_y_vec
        category_new = list()
        for i in range(len(category)):
            if i in n_id:
                category_new.append(category[i])
        with open(document_root_path + 'test_category.pickle', 'wb') as train_collection_file:
            pickle.dump(category_new, train_collection_file, protocol=pickle.HIGHEST_PROTOCOL)
        doc_collection = word_indexing(word2index, doc_collection)
        x_test_tfidf = tfidf_transformer.fit_transform(
            count_vector_constructor(doc_collection, list(index2word.keys())))
        with open(document_root_path + 'test_news_word.pickle', 'wb') as train_file:
            pickle.dump(x_test_tfidf, train_file, protocol=pickle.HIGHEST_PROTOCOL)
    elif level == 1:  # char level
        # handle sentences length
        doc_collection, n_id = handle_docs_length_char_level(doc_collection)
        # remove category of removed doc and add remained one to the  train_y_vec
        category_new = list()
        for i in range(len(category)):
            if i in n_id:
                category_new.append(category[i])
        with open(document_root_path + 'test_category.pickle', 'wb') as train_collection_file:
            pickle.dump(category_new, train_collection_file, protocol=pickle.HIGHEST_PROTOCOL)
        doc_collection = char_indexing(char2index, doc_collection)
        x_test_tfidf = tfidf_transformer.fit_transform(
            count_vector_constructor(doc_collection, list(index2char.keys())))
        with open(document_root_path + 'test_news_char.pickle', 'wb') as train_file:
            pickle.dump(x_test_tfidf, train_file, protocol=pickle.HIGHEST_PROTOCOL)


def evaluation(test_doc_add, index2word, word2index, index2char, char2index, level):
    """
    :param test_doc_add: address of test.csv file
    :param index2word: index2word dictionary
    :param word2index: word2index dictionary
    :param index2char: index2char dictionary
    :param char2index: char2index dictionary
    :param level: training level, an int between 0: word_level and 1: char_level
    :return:

    first call test_doc_prep after that load model based on the level, pass the given test set vectors
     to model to predict a class for them, after that use these prediction and true classes to evaluate model
    """
    if level == 0:  # word level
        y_pred = list()
        y_true = list()
        try:
            with open(document_root_path + 'test_news_word.pickle', 'rb') as train_file:
                pickle.load(train_file)
            with open(document_root_path + 'test_category.pickle', 'rb') as train_collection_file:
                pickle.load(train_collection_file)
        except:
            print('exp 1')
            test_doc_prep(test_doc_add, index2word, word2index, index2char, char2index, level)
            evaluation(test_doc_add, index2word, word2index, index2char, char2index, level)
        # see in predicts hs been saved or not
        try:
            with open(document_root_path + 'test_category.pickle', 'rb') as test_category_file:
                y_true = np.array(pickle.load(test_category_file))
            with open(document_root_path + 'y_predict_word.pickle', 'rb') as y_predict_file:
                y_pred = pickle.load(y_predict_file, encoding='utf8')
        except:
            print('exp2')
            with open(document_root_path + 'SVM_model_word_base.pickle', 'rb') as word_base_model_file:
                word_base_model = pickle.load(word_base_model_file)
            with open(document_root_path + 'test_news_word.pickle', 'rb') as test_news_file:
                x_test = pickle.load(test_news_file)
                y_predict = word_base_model.predict(x_test)
                with open(document_root_path + 'y_predict_word.pickle', 'wb') as y_predict_file:
                    pickle.dump(y_predict, y_predict_file, protocol=pickle.HIGHEST_PROTOCOL)

            evaluation(test_doc_add, index2word, word2index, index2char, char2index, level)

        print('confusion matrix: \n', multilabel_confusion_matrix(y_true, y_pred))

        print('accuracy:    {}'.format(accuracy_score(y_true, y_pred)))

        f1 = f1_score(y_true, y_pred, average=None)
        print('f1 per class:    \n{}'.format(f1))
        print('f1 micro:    {}\n'.format(sum(f1) / 12))

        precision = precision_score(y_true, y_pred, average=None)
        print('precision per class:    \n{}'.format(precision))
        print('precision micro:    {}\n'.format(sum(precision) / 12))
        recall = recall_score(y_true, y_pred, average=None)

        print('recall per class:    \n{}'.format(recall))
        print('recall micro:    {}\n'.format(sum(recall) / 12))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 0
        classes_index = dict()
        with open(document_root_path + 'SVM_model_word_base.pickle', 'rb') as word_base_model_file:
            word_base_model = pickle.load(word_base_model_file)

            with open(document_root_path + 'test_news_word.pickle', 'rb') as test_news_file:
                x_test = pickle.load(test_news_file)
                y_score = word_base_model.decision_function(x_test)
                # Compute ROC curve and ROC area for each class

                with open(document_root_path + 'train_y_vec.pickle', 'rb') as train_category_file:
                    classes = np.unique(np.array(pickle.load(train_category_file, encoding='utf8')))
                    n_classes = len(classes)

                    for i in range(len(classes)):
                        classes_index.update({i: classes[i]})

                y_test = list()
                with open(document_root_path + 'test_category.pickle', 'rb') as test_category_file:
                    y_test_old = pickle.load(test_category_file, encoding='utf8')

                    for cat in y_test_old:
                        ls = list()
                        for cl in classes_index.values():
                            if cat == cl:
                                ls.append(1)
                            else:
                                ls.append(0)
                        y_test.append(ls)
                y_test = np.array(y_test)
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        color = ['darkorange', 'cornflowerblue', 'darkorange', 'aqua', 'seagreen', 'darkcyan', 'slateblue',
                 'lightcoral', 'darkviolet', 'salmon', 'crimson', 'darkolivegreen']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=color[i], lw=lw,
                     label='ROC {} curve (area = %0.2f)'.format(classes_index[i]) % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Word Level')
        plt.legend(loc="lower right")
        plt.show()
    elif level == 1:  # char level
        y_pred = list()
        y_true = list()
        try:
            with open(document_root_path + 'test_news_char.pickle', 'rb') as train_file:
                pickle.load(train_file)
            with open(document_root_path + 'test_category.pickle', 'rb') as train_collection_file:
                pickle.load(train_collection_file)
        except:
            test_doc_prep(test_doc_add, index2word, word2index, index2char, char2index, level)
            evaluation(test_doc_add, index2word, word2index, index2char, char2index, level)

        try:

            with open(document_root_path + 'test_category.pickle', 'rb') as test_category_file:
                y_true = np.array(pickle.load(test_category_file))
                with open(document_root_path + 'y_predict_char.pickle', 'rb') as y_predict_file:
                    y_pred = pickle.load(y_predict_file, encoding='utf8')
        except:
            with open(document_root_path + 'SVM_model_char_base.pickle', 'rb') as char_base_model_file:
                char_base_model = pickle.load(char_base_model_file)
            with open(document_root_path + 'test_news_char.pickle', 'rb') as test_news_file:
                x_test = pickle.load(test_news_file)
                y_predict = char_base_model.predict(x_test)
                with open(document_root_path + 'y_predict_char.pickle', 'wb') as y_predict_file:
                    pickle.dump(y_predict, y_predict_file, protocol=pickle.HIGHEST_PROTOCOL)

            evaluation(test_doc_add, index2word, word2index, index2char, char2index, level)

        print('confusion matrix:    \n', multilabel_confusion_matrix(y_true, y_pred))

        print('accuracy:    {}'.format(accuracy_score(y_true, y_pred)))

        f1 = f1_score(y_true, y_pred, average=None)
        print('f1 per class:    \n{}'.format(f1))
        print('f1 micro:    {}\n'.format(sum(f1) / 12))

        precision = precision_score(y_true, y_pred, average=None)
        print('precision per class:    \n{}'.format(precision))
        print('precision micro:    {}\n'.format(sum(precision) / 12))
        recall = recall_score(y_true, y_pred, average=None)

        print('recall per class:    \n{}'.format(recall))
        print('recall micro:    {}\n'.format(sum(recall) / 12))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 0
        classes_index = dict()
        with open(document_root_path + 'train_y_vec.pickle', 'rb') as train_category_file:
            classes = np.unique(np.array(pickle.load(train_category_file, encoding='utf8')))
            n_classes = len(classes)

            for i in range(len(classes)):
                classes_index.update({i: classes[i]})
        with open(document_root_path + 'SVM_model_char_base.pickle', 'rb') as word_base_model_file:
            word_base_model = pickle.load(word_base_model_file)

            with open(document_root_path + 'test_news_char.pickle', 'rb') as test_news_file:
                x_test = pickle.load(test_news_file)
                y_score = word_base_model.decision_function(x_test)

                y_test = list()
                with open(document_root_path + 'test_category.pickle', 'rb') as test_category_file:
                    y_test_old = pickle.load(test_category_file, encoding='utf8')

                    for cat in y_test_old:
                        ls = list()
                        for cl in classes_index.values():
                            if cat == cl:
                                ls.append(1)
                            else:
                                ls.append(0)
                        y_test.append(ls)
                y_test = np.array(y_test)
                # Compute ROC curve and ROC area for each class
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        color = ['darkorange', 'cornflowerblue', 'darkorange', 'aqua', 'seagreen', 'darkcyan', 'slateblue',
                 'lightcoral', 'darkviolet', 'salmon', 'crimson', 'darkolivegreen']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=color[i], lw=lw,
                     label='ROC {} curve (area = %0.2f)'.format(classes_index[i]) % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Char Level')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    print('evaluation')
    with open(document_root_path + 'char2index.pickle', 'rb') as c2i_file:
        c2i = pickle.load(c2i_file, encoding='utf8')
        with open(document_root_path + 'index2char.pickle', 'rb') as i2c_file:
            i2c = pickle.load(i2c_file, encoding='utf8')
            with open(document_root_path + 'word2index.pickle', 'rb') as w2i_file:
                w2i = pickle.load(w2i_file, encoding='utf8')
                with open(document_root_path + 'index2word.pickle', 'rb') as i2w_file:
                    i2w = pickle.load(i2w_file, encoding='utf8')
                    # calling evaluation function for word_base model
                    evaluation(given_doc_root_path + 'test.csv', i2w, w2i, i2c, c2i, 0)
