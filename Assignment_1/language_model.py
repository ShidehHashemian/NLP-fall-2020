from re import split
import json
from numpy import log10 as log
from first_assignment.LP_toolkits import normalizer, tokenizer
from first_assignment.constants import document_root_path, sentences_avg_length, given_data_root_path
import random
from jiwer import wer


class LanguageModel:

    def __init__(self, n, smoothing, corpus_dir):

        self.n = n
        self.smoothing = smoothing
        self.corpus_dir = corpus_dir
        self.max_prob = {}

    # #     2.1:    implementation of training function (4 below functions)
    def unigram_constructor(self):
        unigram = {}

        with open(self.corpus_dir, 'r', encoding='utf8') as training_corpus_file:
            for line in training_corpus_file:
                line_terms = line.split()

                # #     adding  FOS (First Of Sentence) and LOS (Last Of Sentence) signs to training sentence
                line_terms.insert(0, '<s>')
                line_terms.insert(len(line_terms), '</s>')

                # unigram
                for term in line_terms:

                    if term in unigram.keys():
                        unigram[term] += 1
                    else:
                        unigram.update({term: 1})

        with open(document_root_path + '\\training_unigram.json', 'w', encoding='utf8') as training_unigram_file:
            json.dump(unigram, training_unigram_file, ensure_ascii=False)

        next_word = sorted(unigram.items(), key=lambda item: item[1], reverse=True)[0][0]
        self.max_prob.update({next_word: ''})
        with open(document_root_path + 'training_unigram_prob.json', 'w',
                  encoding='utf8') as training_unigram_prob_file:
            json.dump(self.max_prob, training_unigram_prob_file, ensure_ascii=False)

    def bigram_construction(self):

        # #     {ti:{ti1:freq, ti2:freq, ...}, tj:{tj1:freq, tj2:freq, ...}, ...}
        bigram = {}
        bigram_continuation = {}

        with open(self.corpus_dir, 'r', encoding='utf8') as training_corpus_file:

            for line in training_corpus_file:
                line_terms = line.split()
                # #     adding  FOS (First Of Sentence) and LOS (Last Of Sentence) signs to training sentence
                line_terms.insert(0, '<s>')
                line_terms.insert(len(line_terms), '</s>')

                for i in range(len(line_terms) - 1):
                    ti0 = line_terms[i]
                    ti1 = line_terms[i + 1]

                    # #     create bigram
                    if ti0 in bigram.keys():
                        if ti1 in bigram[ti0].keys():
                            bigram[ti0][ti1] += 1
                        else:
                            bigram[ti0].update({ti1: 1})
                    else:
                        bigram.update({ti0: {ti1: 1}})
                    # #     create bigram_continuation to for further use in kneser-ney smoothing
                    if self.smoothing == 'kneser-ney':
                        if ti1 in bigram_continuation.keys():
                            if not (ti0 in bigram_continuation[ti1]):
                                bigram_continuation[ti1].append(ti0)
                        else:
                            bigram_continuation.update({ti1: [ti0]})

        with open(document_root_path + '\\training_bigram.json', 'w', encoding='utf8') as training_bigram_file:
            json.dump(bigram, training_bigram_file, ensure_ascii=False)

        for term in bigram.keys():
            sorted_frequency = sorted(bigram[term].items(), key=lambda item: item[1], reverse=True)
            self.max_prob.update({term: sorted_frequency[0][0]})
        with open(document_root_path + 'training_bigram_prob.json', 'w',
                  encoding='utf8') as training_bigram_prob_file:
            json.dump(self.max_prob, training_bigram_prob_file, ensure_ascii=False)

        with open(document_root_path + 'training_bigram_continuation.json', 'w',
                  encoding='utf8') as training_bigram_continuation_file:
            json.dump(bigram_continuation, training_bigram_continuation_file, ensure_ascii=False)

    def trigram_construction(self):

        # #     {ti:{ti1:{ti11:freq,ti12:freq, ...}, ...}, tj:tj:{tj1:{tj11:freq,tj12:freq, ...}, ...}, ...}
        trigram = {}
        trigram_continuation = {}
        with open(self.corpus_dir, 'r', encoding='utf8') as training_corpus_file:
            for line in training_corpus_file:
                line_terms = line.split()
                # #     adding  FOS (First Of Sentence) and LOS (Last Of Sentence) signs to training sentence
                line_terms.insert(0, '<s>')
                line_terms.insert(len(line_terms), '</s>')

                for i in range(len(line_terms) - 2):
                    ti0 = line_terms[i]
                    ti1 = line_terms[i + 1]
                    ti2 = line_terms[i + 2]

                    if ti0 in trigram.keys():
                        if ti1 in trigram[ti0].keys():
                            if ti2 in trigram[ti0][ti1].keys():
                                trigram[ti0][ti1][ti2] += 1
                            else:
                                trigram[ti0][ti1].update({ti2: 1})
                        else:
                            trigram[ti0].update({ti1: {ti2: 1}})
                    else:
                        trigram.update({ti0: {ti1: {ti2: 1}}})

                    # #     create trigram_continuation to for further use in kneser-ney smoothing
                    if ti2 in trigram_continuation.keys():
                        if ti1 in trigram_continuation[ti2].keys():
                            if not (ti0 in trigram_continuation[ti2][ti1]):
                                trigram_continuation[ti2][ti1].append(ti0)
                        else:
                            trigram_continuation[ti2].update({ti1: [ti0]})
                    else:
                        trigram_continuation.update({ti2: {ti1: [ti0]}})

            with open(document_root_path + 'training_trigram.json', 'w', encoding='utf8') as training_trigram_file:
                json.dump(trigram, training_trigram_file, ensure_ascii=False)

            for t0 in trigram.keys():
                for t1 in trigram[t0].keys():
                    most_probable = sorted(trigram[t0][t1].items(), key=lambda item: item[1], reverse=True)[0][0]

                    if t0 in self.max_prob.keys():
                        self.max_prob[t0].update({t1: most_probable})
                    else:
                        self.max_prob.update({t0: {t1: most_probable}})

            with open(document_root_path + 'training_trigram_prob.json', 'w',
                      encoding='utf8') as training_trigram_prob_file:
                json.dump(self.max_prob, training_trigram_prob_file, ensure_ascii=False)

            with open(document_root_path + 'training_trigram_continuation.json', 'w',
                      encoding='utf8') as training_trigram_continuation_file:
                json.dump(trigram_continuation, training_trigram_continuation_file, ensure_ascii=False)

    def train(self):
        if self.n == 0:
            self.unigram_constructor()
        elif self.n == 1:
            self.unigram_constructor()
            self.bigram_construction()
        elif self.n == 2:
            self.bigram_construction()
            self.trigram_construction()

        # #     2.2:    implementation of prob function (4 below functions)

    def unigram_prob(self, sentence):
        sentence_terms = sentence.split()
        sentence_terms.insert(0, '<s>')
        sentence_terms.insert(len(sentence_terms), '</s>')

        with open(document_root_path + 'training_unigram.json', 'r', encoding='utf8') as training_unigram_file:
            unigram = json.load(training_unigram_file)

            total_token = sum(unigram.values())
            p_sentence = 0.

            for word in sentence_terms:
                if word in unigram.keys():

                    if self.smoothing == 'laplace':
                        p_sentence += log(float(unigram[word] + 1) / float(total_token + len(unigram) - 2))

                    elif not self.smoothing:
                        p_sentence += log(float(unigram[word]) / float(total_token))
                else:
                    if self.smoothing == 'laplace':
                        p_sentence += log(float(1) / float(total_token + len(unigram) - 2))

            print('p (\'' + sentence + '\')=     ', p_sentence)

    def bigram_prob(self, sentence):
        sentence_terms = sentence.split()
        sentence_terms.insert(0, '<s>')
        sentence_terms.insert(len(sentence_terms), '</s>')

        with open(document_root_path + 'training_bigram.json', 'r', encoding='utf8') as training_bigram_file:
            bigram = json.load(training_bigram_file)

            with open(document_root_path + 'training_unigram.json', 'r', encoding='utf8') as training_unigram_file:
                unigram = json.load(training_unigram_file)

                p_sentence = 0.
                if not self.smoothing:
                    for i in range(len(sentence_terms) - 1):

                        wi0 = sentence_terms[i]
                        wi1 = sentence_terms[i + 1]

                        if wi0 in bigram.keys():
                            if wi1 in bigram[wi0].keys():
                                p_sentence += log(float(bigram[wi0][wi1]) / float(unigram[wi0]))
                            else:
                                p_sentence += float('-inf')
                        else:
                            p_sentence += float('-inf')
                elif self.smoothing == 'laplace':
                    for i in range(len(sentence_terms) - 1):

                        wi0 = sentence_terms[i]
                        wi1 = sentence_terms[i + 1]

                        if wi0 in bigram.keys():
                            if wi1 in bigram[wi0].keys():
                                p_sentence += log(
                                    float(bigram[wi0][wi1] + 1) / float(unigram[wi0] + len(unigram.keys()) - 2))
                            else:
                                p_sentence += log(float(1) / float(unigram[wi0] + len(unigram.keys()) - 2))
                        else:
                            p_sentence += log(float(1) / float(len(unigram.keys()) - 2))
                elif self.smoothing == 'kneser-ney':
                    # calculated based n given formula in chapter3 of the source book kneser-ney description

                    with open(document_root_path + 'training_bigram_continuation.json', 'r',
                              encoding='utf8') as training_bigram_continuation_file:
                        bigram_continuation = json.load(training_bigram_continuation_file)

                        total_unique = 0
                        for wi0 in bigram_continuation.keys():
                            total_unique += len(bigram_continuation[wi0])
                        for i in range(len(sentence_terms) - 1):

                            wi0 = sentence_terms[i]
                            wi1 = sentence_terms[i + 1]

                            if wi0 in bigram.keys():
                                if wi1 in bigram[wi0].keys():

                                    if wi1 in bigram_continuation.keys():
                                        c_wi0_wi1 = float(bigram[wi0][wi1])
                                        c_wi0 = float(unigram[wi0])
                                        lambda_wi0 = (0.75 / c_wi0) * float(len(bigram[wi0]))
                                        pc_wi1 = float(len(bigram_continuation[wi1]) / total_unique)
                                        p_wi1_wi0 = (max(c_wi0_wi1 - 0.75, 0.) / c_wi0) + lambda_wi0 * pc_wi1
                                        if p_wi1_wi0 > 0:
                                            p_sentence += log(p_wi1_wi0)
                                        else:
                                            p_sentence += float('-inf')
                                    else:
                                        c_wi0_wi1 = float(bigram[wi0][wi1])
                                        c_wi0 = float(unigram[wi0])
                                        p_wi1_wi0 = (max(c_wi0_wi1 - 0.75, 0.) / c_wi0)
                                        if p_wi1_wi0 > 0:
                                            p_sentence += log(p_wi1_wi0)
                                        else:
                                            p_sentence += float('-inf')
                                else:
                                    if wi1 in bigram_continuation.keys():
                                        c_wi0_wi1 = float(0)
                                        c_wi0 = float(unigram[wi0])
                                        lambda_wi0 = (0.75 / c_wi0) * float(len(bigram[wi0]))
                                        pc_wi1 = float(len(bigram_continuation[wi1]) / total_unique)
                                        p_wi1_wi0 = (max(c_wi0_wi1 - 0.75, 0.) / c_wi0) + lambda_wi0 * pc_wi1
                                        if p_wi1_wi0 > 0:
                                            p_sentence += log(p_wi1_wi0)
                                        else:
                                            p_sentence += float('-inf')
                                    else:
                                        p_sentence += float('-inf')
                            else:
                                p_sentence += float('-inf')

                print('p (\'' + sentence + '\')=     ', p_sentence)

    def trigram_prob(self, sentence):
        sentence_terms = sentence.split()
        sentence_terms.insert(0, '<s>')
        sentence_terms.insert(len(sentence_terms) + 1, '</s>')

        with open(document_root_path + 'training_bigram.json', 'r', encoding='utf8') as training_bigram_file:
            bigram = json.load(training_bigram_file)

            with open(document_root_path + 'training_trigram.json', 'r', encoding='utf8') as training_trigram_file:
                trigram = json.load(training_trigram_file)

                if not self.smoothing:
                    for i in range(len(sentence_terms) - 2):
                        wi0 = sentence_terms[i]
                        wi1 = sentence_terms[i + 1]
                        wi2 = sentence_terms[i + 2]

                        p_sentence = 0.
                        if wi0 in trigram.keys():
                            if wi1 in trigram[wi0].keys():
                                if wi2 in trigram[wi0][wi1].keys():
                                    p_sentence += log(float(trigram[wi0][wi1][wi2]) / float(bigram[wi0][wi1]))
                                else:
                                    # because the numerator of the likelihood
                                    # fraction would be zero and log(0) would be infinite negative
                                    p_sentence += float('-inf')
                            else:
                                #     because in the likelihood fraction both numerator and denominator are zero,
                                #     which imply the concept that the possibility of this sentence happening
                                #     given the training set is zero
                                #     therefore used the '-inf' to show this concept!
                                p_sentence += float('-inf')
                        else:
                            #     because in the likelihood fraction both numerator and denominator are zero,
                            #     which imply the concept that the possibility of this sentence happening
                            #     given the training set is zero
                            #     therefore used the '-inf' to show this concept!
                            p_sentence += float('-inf')
                elif self.smoothing == 'laplace':
                    for i in range(len(sentence_terms) - 2):
                        wi0 = sentence_terms[i]
                        wi1 = sentence_terms[i + 1]
                        wi2 = sentence_terms[i + 2]

                        p_sentence = 0.
                        if wi0 in trigram.keys():
                            if wi1 in trigram[wi0].keys():
                                if wi2 in trigram[wi0][wi1].keys():
                                    p_sentence += log(float(trigram[wi0][wi1][wi2] + 1) / float(
                                        bigram[wi0][wi1] + len(bigram.keys()) - 2))
                                else:
                                    p_sentence += log(float(1) / float(
                                        bigram[wi0][wi1] + len(bigram.keys()) - 2))
                            else:
                                p_sentence += log(float(1) / float(len(bigram.keys()) - 2))
                        else:
                            p_sentence += log(float(1) / float(len(bigram.keys()) - 2))
                elif self.smoothing == 'kneser-ney':
                    # calculated based n given formula in chapter3 of the source book kneser-ney description
                    with open(document_root_path + 'training_bigram_continuation.json', 'r',
                              encoding='utf8') as training_bigram_continuation_file:
                        bigram_continuation = json.load(training_bigram_continuation_file)

                        total_unique = 0
                        for wi0 in bigram_continuation.keys():
                            total_unique += len(bigram_continuation[wi0])
                        with open(document_root_path + 'training_trigram_continuation.json', 'r',
                                  encoding='utf8') as training_trigram_continuation_file:
                            trigram_continuation = json.load(training_trigram_continuation_file)

                            for i in range(len(sentence_terms) - 2):
                                wi0 = sentence_terms[i]
                                wi1 = sentence_terms[i + 1]
                                wi2 = sentence_terms[i + 2]

                                p_sentence = 0.
                                if wi0 in trigram.keys():
                                    if wi1 in trigram[wi0].keys():
                                        if wi2 in trigram[wi0][wi1].keys():

                                            if wi2 in trigram_continuation.keys():
                                                if wi1 in trigram_continuation[wi2].keys():

                                                    ckn_wi1_wi2 = float(len(trigram_continuation[wi2][wi1]))
                                                    ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                    lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                    pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                    pkn_wi2_wi1 = (max(ckn_wi1_wi2 - 0.75,
                                                                       0.) / ckn_wi1) + lambda_wi1 * pc_wi2

                                                    c_wi0_wi1_wi2 = float(trigram[wi0][wi1][wi2])
                                                    c_wi0_wi1 = float(bigram[wi0][wi1])
                                                    if wi1 in trigram_continuation.keys():
                                                        if wi0 in trigram_continuation[wi1].keys():

                                                            ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                            lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                                len(trigram[wi0][wi1]))
                                                            pkn_wi2_wi1_wi0 = (max(c_wi0_wi1_wi2 - 0.75,
                                                                                   0.) / c_wi0_wi1) + lambda_wi0_wi1 * pkn_wi2_wi1

                                                            if pkn_wi2_wi1_wi0 > 0:
                                                                p_sentence += log(pkn_wi2_wi1_wi0)
                                                            else:
                                                                p_sentence += float('-inf')
                                                        else:  # wi0  not in trigram_continuation[wi1].keys():
                                                            p_sentence += float('-inf')
                                                    else:  # wi1 not in trigram_continuation.keys()
                                                        p_sentence += float('-inf')
                                                else:  # wi1 not in trigram_continuation[wi2].keys()
                                                    ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                    lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                    pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                    pkn_wi2_wi1 = lambda_wi1 * pc_wi2

                                                    c_wi0_wi1_wi2 = float(trigram[wi0][wi1][wi2])
                                                    c_wi0_wi1 = float(bigram[wi0][wi1])
                                                    if wi1 in trigram_continuation.keys():
                                                        if wi0 in trigram_continuation[wi1].keys():

                                                            ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                            lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                                len(trigram[wi0][wi1]))
                                                            pkn_wi2_wi1_wi0 = (max(c_wi0_wi1_wi2 - 0.75,
                                                                                   0.) / c_wi0_wi1) + lambda_wi0_wi1 * pkn_wi2_wi1

                                                            if pkn_wi2_wi1_wi0 > 0:
                                                                p_sentence += log(pkn_wi2_wi1_wi0)
                                                            else:
                                                                p_sentence += float('-inf')
                                                        else:  # wi0  not in trigram_continuation[wi1].keys():
                                                            p_sentence += float('-inf')
                                                    else:  # wi1 not in trigram_continuation.keys()
                                                        p_sentence += float('-inf')
                                            else:  # wi2  not in trigram_continuation.keys():
                                                ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                pkn_wi2_wi1 = lambda_wi1 * pc_wi2

                                                c_wi0_wi1_wi2 = float(trigram[wi0][wi1][wi2])
                                                c_wi0_wi1 = float(bigram[wi0][wi1])
                                                if wi1 in trigram_continuation.keys():
                                                    if wi0 in trigram_continuation[wi1].keys():

                                                        ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                        lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                            len(trigram[wi0][wi1]))
                                                        pkn_wi2_wi1_wi0 = (max(c_wi0_wi1_wi2 - 0.75,
                                                                               0.) / c_wi0_wi1) + lambda_wi0_wi1 * pkn_wi2_wi1

                                                        if pkn_wi2_wi1_wi0 > 0:
                                                            p_sentence += log(pkn_wi2_wi1_wi0)
                                                        else:
                                                            p_sentence += float('-inf')
                                                    else:  # wi0  not in trigram_continuation[wi1].keys():
                                                        p_sentence += float('-inf')
                                                else:  # wi1 not in trigram_continuation.keys()
                                                    p_sentence += float('-inf')
                                        else:  # wi2 not in trigram[wi0][wi1].keys()
                                            if wi2 in trigram_continuation.keys():
                                                if wi1 in trigram_continuation[wi2].keys():

                                                    ckn_wi1_wi2 = float(len(trigram_continuation[wi2][wi1]))
                                                    ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                    lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                    pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                    pkn_wi2_wi1 = (max(ckn_wi1_wi2 - 0.75,
                                                                       0.) / ckn_wi1) + lambda_wi1 * pc_wi2
                                                    if wi1 in trigram_continuation.keys():
                                                        if wi0 in trigram_continuation[wi1].keys():

                                                            ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                            lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                                len(trigram[wi0][wi1]))
                                                            pkn_wi2_wi1_wi0 = lambda_wi0_wi1 * pkn_wi2_wi1

                                                            if pkn_wi2_wi1_wi0 > 0:
                                                                p_sentence += log(pkn_wi2_wi1_wi0)
                                                            else:
                                                                p_sentence += float('-inf')
                                                        else:  # wi0  not in trigram_continuation[wi1].keys():
                                                            p_sentence += float('-inf')
                                                    else:  # wi1 not in trigram_continuation.keys()
                                                        p_sentence += float('-inf')
                                                else:  # wi1 not in trigram_continuation[wi2].keys()
                                                    ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                    lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                    pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                    pkn_wi2_wi1 = lambda_wi1 * pc_wi2

                                                    c_wi0_wi1_wi2 = float(trigram[wi0][wi1][wi2])
                                                    c_wi0_wi1 = float(bigram[wi0][wi1])
                                                    if wi1 in trigram_continuation.keys():
                                                        if wi0 in trigram_continuation[wi1].keys():

                                                            ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                            lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                                len(trigram[wi0][wi1]))
                                                            pkn_wi2_wi1_wi0 = (max(c_wi0_wi1_wi2 - 0.75,
                                                                                   0.) / c_wi0_wi1) + lambda_wi0_wi1 * pkn_wi2_wi1

                                                            if pkn_wi2_wi1_wi0 > 0:
                                                                p_sentence += log(pkn_wi2_wi1_wi0)
                                                            else:
                                                                p_sentence += float('-inf')
                                                        else:  # wi0  not in trigram_continuation[wi1].keys():
                                                            p_sentence += float('-inf')
                                                    else:  # wi1 not in trigram_continuation.keys()
                                                        p_sentence += float('-inf')
                                            else:  # wi2  not in trigram_continuation.keys():
                                                ckn_wi1 = float(len(bigram_continuation[wi1]))
                                                lambda_wi1 = (0.75 / ckn_wi1) * float(len(bigram[wi1]))
                                                pc_wi2 = float(len(bigram_continuation[wi2]) / total_unique)
                                                pkn_wi2_wi1 = lambda_wi1 * pc_wi2

                                                c_wi0_wi1_wi2 = float(trigram[wi0][wi1][wi2])
                                                c_wi0_wi1 = float(bigram[wi0][wi1])
                                                if wi1 in trigram_continuation.keys():
                                                    if wi0 in trigram_continuation[wi1].keys():

                                                        ckn_wi0_wi1 = float(len(trigram_continuation[wi1][wi0]))
                                                        lambda_wi0_wi1 = (0.75 / ckn_wi0_wi1) * float(
                                                            len(trigram[wi0][wi1]))
                                                        pkn_wi2_wi1_wi0 = (max(c_wi0_wi1_wi2 - 0.75,
                                                                               0.) / c_wi0_wi1) + lambda_wi0_wi1 * pkn_wi2_wi1

                                                        if pkn_wi2_wi1_wi0 > 0:
                                                            p_sentence += log(pkn_wi2_wi1_wi0)
                                                        else:
                                                            p_sentence += float('-inf')
                                                    else:  # wi0  not in trigram_continuation[wi1].keys():
                                                        p_sentence += float('-inf')
                                                else:  # wi1 not in trigram_continuation.keys()
                                                    p_sentence += float('-inf')
                                    else:  # wi1 not in trigram[wi0].keys()
                                        p_sentence += float('-inf')
                                else:  # wi0 in trigram.keys()
                                    p_sentence += float('-inf')

            print('p (\'' + sentence + '\')=     ', p_sentence)

    def prob(self, sentence):
        if self.n == 0:
            self.unigram_prob(sentence)
        elif self.n == 1:
            self.bigram_prob(sentence)
        elif self.n == 2:
            self.trigram_prob(sentence)

    # #     2.3:    implementation of generator function (4 below functions)

    def unigram_generator(self, sentence):
        return list(self.max_prob.keys())[0]

    def bigram_generator(self, sentence):
        sentence_terms = sentence.split()
        sentence_terms.insert(0, '<s>')
        next_word = ''
        if sentence_terms[-1] in self.max_prob.keys():

            next_word = self.max_prob[sentence_terms[-1]]
            return next_word
        else:
            if self.smoothing == 'laplace':
                index = random.randint(0, len(self.max_prob) - 1)
                next_word = list(self.max_prob.keys())[index]
                return next_word
            else:
                # #     as for both none smoothing applied adn kneser-smoothing,
                # if the previous word is not in the vocabulary, the given probability would be zero or infinite
                # so no word would be suggested
                return next_word

    def trigram_generator(self, sentence):
        sentence_terms = sentence.split()
        sentence_terms.insert(0, '<s>')

        next_word = ''

        if sentence_terms[-2] in self.max_prob.keys():
            if sentence_terms[-1] in self.max_prob[sentence_terms[-2]].keys():

                next_word = self.max_prob[sentence_terms[-2]][sentence_terms[-1]]
                return next_word
            else:
                if self.smoothing == 'laplace':
                    index = random.randint(0, len(self.max_prob) - 1)
                    next_word = list(self.max_prob.keys())[index]
                    return next_word
                elif not self.smoothing:
                    return next_word
        else:
            if self.smoothing == 'laplace':
                index = random.randint(0, len(self.max_prob) - 1)
                next_word = list(self.max_prob.keys())[index]
                return next_word
            elif not self.smoothing:
                return next_word

    def generate(self, sentence):
        if self.n == 0:
            with open(document_root_path + 'training_unigram_prob.json', 'r',
                      encoding='utf8') as training_unigram_prob_file:
                self.max_prob = json.load(training_unigram_prob_file)
            next_word = self.unigram_generator(sentence)
            if len(next_word) > 0:
                print(sentence, next_word)
            else:
                print('sorry, the trained model couldn\'t come up with any suggestion for next word!')
        elif self.n == 1:
            with open(document_root_path + 'training_bigram_prob.json', 'r',
                      encoding='utf8') as training_bigram_prob_file:
                self.max_prob = json.load(training_bigram_prob_file)
            next_word = self.unigram_generator(sentence)
            if len(next_word) > 0:
                print(sentence, next_word)
            else:
                print('sorry, the trained model couldn\'t come up with any suggestion for next word!')
        elif self.n == 2:
            with open(document_root_path + 'training_trigram_prob.json', 'r',
                      encoding='utf8') as training_trigram_prob_file:
                self.max_prob = json.load(training_trigram_prob_file)
            next_word = self.unigram_generator(sentence)
            if len(next_word) > 0:
                print(sentence, next_word)
            else:
                print('sorry, the trained model couldn\'t come up with any suggestion for next word!')

    # #     3:    implementation of evaluation function (4 below functions)
    def evaluation(self, valid_corpus_dir):
        dictionary = []
        total_sentences = 0
        total_wer = 0

        if self.n == 0:
            with open(document_root_path + 'training_unigram_prob.json', 'r',
                      encoding='utf8') as training_unigram_prob_file:
                self.max_prob = json.load(training_unigram_prob_file)
        elif self.n == 1:

            with open(document_root_path + 'training_bigram_prob.json', 'r',
                      encoding='utf8') as training_bigram_prob_file:
                self.max_prob = json.load(training_bigram_prob_file)
        elif self.n == 2:
            with open(document_root_path + 'training_trigram_prob.json', 'r',
                      encoding='utf8') as training_trigram_prob_file:
                self.max_prob = json.load(training_trigram_prob_file)

        with open(document_root_path + 'most_frequent.txt', encoding='utf8') as most_frequent_file:
            most_frequent_word_read = most_frequent_file.read()
            for line in most_frequent_word_read.split():
                dictionary.append(line.split()[0])

        with open(valid_corpus_dir, 'r', encoding='utf8') as training_corpus_file:

            if self.n == 0:
                for line in training_corpus_file:
                    total_sentences += 1
                    line_terms = line.split()
                    hypothesis_sentence = '<s> '
                    if not line_terms[0] in dictionary:
                        hypothesis_sentence += ' ' + self.unigram_generator(hypothesis_sentence)
                    else:
                        hypothesis_sentence += line_terms[0]
                    counter = 1
                    while counter < sentences_avg_length and hypothesis_sentence.split()[-1] != '</s>':
                        counter += 1
                        hypothesis_sentence += ' ' + self.unigram_generator(hypothesis_sentence)
                    line_terms = line.split()
                    line_terms.insert(0, '<s>')
                    line_terms.insert(len(line_terms), '</s>')

                    total_wer += wer(' '.join(line_terms), hypothesis_sentence)

            elif self.n == 1:
                for line in training_corpus_file:
                    total_sentences += 1
                    line_terms = line.split()
                    hypothesis_sentence = '<s> '
                    if not line_terms[0] in dictionary:
                        hypothesis_sentence += ' ' + self.bigram_generator(hypothesis_sentence)
                    else:
                        hypothesis_sentence += line_terms[0]
                    counter = 1
                    while counter < sentences_avg_length and hypothesis_sentence.split()[-1] != '</s>':
                        counter += 1
                        hypothesis_sentence += ' ' + self.bigram_generator(hypothesis_sentence)
                    line_terms = line.split()
                    line_terms.insert(0, '<s>')
                    line_terms.insert(len(line_terms), '</s>')

                    total_wer += wer(' '.join(line_terms), hypothesis_sentence)

            elif self.n == 2:
                for line in training_corpus_file:
                    total_sentences += 1
                    line_terms = line.split()
                    hypothesis_sentence = '<s> '
                    if not line_terms[0] in dictionary:

                        hypothesis_sentence += ' ' + self.trigram_generator(hypothesis_sentence)
                    else:
                        hypothesis_sentence += line_terms[0]
                    counter = 1
                    while counter < sentences_avg_length and hypothesis_sentence.split()[-1] != '</s>':
                        counter += 1
                        hypothesis_sentence += ' ' + self.trigram_generator(hypothesis_sentence)
                    line_terms = line.split()
                    line_terms.insert(0, '<s>')
                    line_terms.insert(len(line_terms), '</s>')

                    total_wer += wer(' '.join(line_terms), hypothesis_sentence)

        print(total_wer / total_sentences)


def evaluation_corpus_prep(raw_valid_corpus_dir):
    most_frequent_word = []
    with open(document_root_path + 'most_frequent.txt', encoding='utf8') as most_frequent_file:

        most_frequent_word_read = most_frequent_file.read()
        for line in most_frequent_word_read.split():
            most_frequent_word.append(line.split()[0])

    with open(raw_valid_corpus_dir, encoding='utf8') as validation_corpus_file:
        corpus = json.load(validation_corpus_file)
        with open(document_root_path + 'valid_sentences.txt', 'w', encoding='utf8') as vali_sentences_file:

            for each in corpus:
                news = normalizer(each)
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

                sentences = split('[.؟?]', ' '.join(news_array))
                for sentence in sentences:
                    if not sentence == ' ' and len(sentence) > 0:
                        vali_sentences_file.write(sentence + '\n')


if __name__ == '__main__':
    # to prepare the validation data to be used in other parts, and after runing tthis function it'll
    # save a normalized validation txt file named valid_sentences.txt
    evaluation_corpus_prep(given_data_root_path)

    # below lines show how to use this class, and by changing the passed value based on what they mean
    # (which are mentioned in the assignment instruction) you can construct other language models

    # language model with use unigram and no smoothing method
    lm_unigram_none_smoothing = LanguageModel(0, False, document_root_path + '\\train_sentences.txt')
    # train mentioned language model
    lm_unigram_none_smoothing.train()
    # calculate the probability of the given sentence to happen
    lm_unigram_none_smoothing.prob('به گزارش')
    # generate hypothetical next word
    lm_unigram_none_smoothing.generate('به گزارش')
    # validate the trained model using processed given validation file in the assignment
    lm_unigram_none_smoothing.evaluation(document_root_path + 'valid_sentences.txt')
