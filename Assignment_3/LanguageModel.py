import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
import pickle
from constants import document_root_path, given_doc_root_path
from LP_toolkits import normalizer
import re
import pandas as pd


# define the same class for RNN to use it to load the model
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


class LanguageModel:

    def __init__(self, lm_checkpoints, char2index_path, index2char_path):
        """
        :param lm_checkpoints: path to models weight file
        :param char2index_path: path to char2index.pickle file
        :param index2char_path:path to index2char.pickle file
        """
        self.LSTM = None

        with open(char2index_path, 'rb') as char2index_file:
            self.char2index = pickle.load(char2index_file, encoding='utf8')
        with open(index2char_path, 'rb') as index2char_file:
            self.index2char = pickle.load(index2char_file, encoding='utf8')
        self.vocab_size = len(self.char2index)
        self.lm_unit(weights=lm_checkpoints)

    def lm_unit(self, ininti_state=None, weights=None):
        """
        :param ininti_state: its not needed if the model has first hidden zero layer
        :param weights: weights of the language model
        :return: the trained language model
        """

        # ##########  Hyperparameters  ###########
        hidden_size = 256  # size of hidden state
        num_layers = 2  # num of layers in LSTM layer stack
        # ######################################

        # create and load model instance
        self.LSTM = RNN(self.vocab_size, self.vocab_size, hidden_size, num_layers)
        self.LSTM.load_state_dict(torch.load(weights))
        print("Model loaded successfully !!")

    def get_next_state_and_output(self, prefix):
        """

        :param prefix: a string
        :return: an array of chars probability and a tuple of (hn,cn)
        """
        # as we don't use the model for training, call the eval mode
        self.LSTM.eval()
        # add start sign to the first of the input and normalize it to eliminate redundant chars
        prefix = '\\n' + normalizer(prefix)
        # use char2index dict to index the input string
        str_index = [self.char2index[ch] for ch in prefix[2:]]
        str_index.insert(0, self.char2index['\\n'])
        str_len = len(str_index)
        # convert it to torch and change its dim to use it as models input
        str_index = torch.tensor(str_index)
        str_index = torch.unsqueeze(str_index, dim=1)
        # initial hidden state
        hidden_state = None
        # compute last hidden state of the sequence
        _, hidden_state = self.LSTM(str_index[:-1], hidden_state)
        #  forward pass
        output, hidden_state = self.LSTM(str_index[str_len - 1:str_len], hidden_state)

        #  get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving prefix
        output = F.softmax(torch.squeeze(output), dim=0)

        return output, hidden_state

    def prefix_to_hidden(self, prefix):
        """

        :param prefix: a string
        :return: hidden layers vectors (here would be c and h)
        """
        # as we don't use the model for training, call the eval mode
        self.LSTM.eval()
        # normalize input prefix
        sentence = normalizer(prefix)
        # index prefix and also add '\\n' to the first of it
        str_index = [self.char2index[ch] for ch in sentence]
        str_index.insert(0, self.char2index['\\n'])
        hidden_state = None

        # compute last hidden state of the sequence

        _, hidden_state = self.LSTM(torch.unsqueeze(torch.tensor(str_index), dim=1), hidden_state)
        return hidden_state

    def generate_new_sample(self, prefix):
        """

        :param prefix: a string
        :return:  a sentence generated_using this prefix
        """
        # as we don't use the model for training, call the eval mode
        self.LSTM.eval()
        # normalize input prefix
        sentence = normalizer(prefix)
        # index prefix and also add '\\n' to the first of it
        str_index = [self.char2index[ch] for ch in sentence]
        str_index.insert(0, self.char2index['\\n'])
        n = len(str_index)
        hidden_state = None

        # compute last hidden state of the sequence
        if n > 1:
            _, hidden_state = self.LSTM(torch.unsqueeze(torch.tensor(str_index), dim=1)[:n - 1], hidden_state)

        # generate sentence which ends either with '\\t\' or it's generated 280 chars
        # (somewhere near avg sentences max chars)
        for i in range(280):

            output, hidden_state = self.LSTM(torch.unsqueeze(torch.tensor(str_index), dim=1)[n - 1:n], hidden_state)

            #  get the character probabilities
            # apply softmax to get p probabilities for the likely next character giving previous char
            output = F.softmax(torch.squeeze(output), dim=0).data

            # use top k sampling which allowing the model to introduce some noise and randomness into the sampled text
            # randomly (more chance for index with higher prob) choose an index
            top_ch = np.arange(len(list(self.char2index.keys())))
            p = output.numpy().squeeze()
            index = np.random.choice(top_ch, p=p / p.sum())

            # check if the sentence is finished or not (as '\\t' is a sign of it)
            if index == self.char2index['\\t']:
                return sentence

            # add the new char to the sentence
            sentence += self.index2char[index]
            # add it's index to str_index to use it for generating new char
            str_index.append(index)
            n = len(str_index)
        return sentence

    def get_probability(self, prefix, total=False):
        """

        :param total: True if we want the array of probabilities, False if we only want the max probability
        :param prefix:  a vector of  prefix
        :return: a float which is the probability of the chosen char by the model
        """
        # as we don't use the model for training, call the eval mode
        self.LSTM.eval()

        # normalize input prefix
        sentence = normalizer(prefix)
        # index prefix and also add '\\n' to the first of it
        str_index = [self.char2index[ch] for ch in sentence]
        str_index.insert(0, self.char2index['\\n'])

        n = len(str_index)
        # change its type to be able to use it as an input of the model
        str_index = torch.unsqueeze(torch.tensor(str_index), dim=1)

        hidden_state = None
        # compute last hidden state of the sequence
        if n > 1:
            _, hidden_state = self.LSTM(str_index[:n - 1], hidden_state)
        output, hidden_state = self.LSTM(str_index[n - 1:n], hidden_state)

        # get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving prefix
        output = F.softmax(torch.squeeze(output), dim=0).data
        if total:  # use this one in get_overall_probability function
            return output
        else:
            return self.index2char[int(torch.argmax(output))], float(torch.max(output))

    def get_overall_probability(self, full_sentence):
        """

        :param full_sentence: get a full sentence as a string
        :return: log sentence probability

        """
        # as we don't use the model for training, call the eval mode
        self.LSTM.eval()

        total_prob_lg = 0
        # normalize full sentence to be able to get te index of each char
        full_sentence = normalizer(full_sentence)
        n = 0
        while n < len(full_sentence):
            output = self.get_probability(full_sentence[:n], total=True)
            if float(output[self.char2index[full_sentence[n]]]) != 0:
                total_prob_lg += np.log10(float(output[self.char2index[full_sentence[n]]]))
            n += 1
        # includes full sentence output (which should be '\\t')
        output = self.get_probability(full_sentence[:n], total=True)
        if float(output[self.char2index['\\t']]) != 0:
            total_prob_lg += np.log10(float(output[self.char2index['\\t']]))
        return total_prob_lg

    def evaluation(self, full_sentence):

        def cer(r, h):
            # initialisation

            d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
            d = d.reshape((len(r) + 1, len(h) + 1))
            for i in range(len(r) + 1):
                for j in range(len(h) + 1):
                    if i == 0:
                        d[0][j] = j
                    elif j == 0:
                        d[i][0] = i

            # computation
            for i in range(1, len(r) + 1):
                for j in range(1, len(h) + 1):
                    if r[i - 1] == h[j - 1]:
                        d[i][j] = d[i - 1][j - 1]
                    else:
                        substitution = d[i - 1][j - 1] + 1
                        insertion = d[i][j - 1] + 1
                        deletion = d[i - 1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)

            return d[len(r)][len(h)]

        result = {'CER': float,
                  'Perplexity': float}
        full_sentence = normalizer(full_sentence)
        prefix = full_sentence.split()[0]
        predicted_sentence = self.generate_new_sample(prefix)
        result['CER'] = cer(full_sentence, predicted_sentence)

        # use self.get_overall_probability to get the probability of the generated sentence a
        # nd use it to compute perplexity
        prob = self.get_overall_probability(predicted_sentence)
        result['Perplexity'] = np.power((np.power(prob, -10)), 1 / (len(predicted_sentence) + 1))
        return result


def generate_sample_using_test_file(test_file_path, model_path):
    def read_csv_data(doc_add):
        csv_data = pd.read_csv(doc_add, delimiter='\t')

        doc_sentences = []
        doc_id = 0

        for index in range(csv_data.shape[0]):
            # as normalizer remove last space, add it here
            text = normalizer(str(csv_data['text'][index])) + ' '
            sentences = sentence_formation(text)
            for sen in sentences:
                doc_sentences.append(sen)
            doc_id += 1
            if doc_id % 10000 == 0:
                print(doc_id)
        print('####   done reading file   ####')
        return doc_sentences

    def sentence_formation(news):
        sentences = list()
        end_points = sorted([r.start() for r in re.finditer('[.؟]', news)])
        start_index = 0
        for i in end_points:
            if str(news[start_index:i]) != ' ' and len(str(news[start_index:i])) != 0:
                sentences.append('\\n' + news[start_index:i + 2] + '\\t')
                start_index = i + 2
        return sentences

    avg_cer = 0.
    avg_perplexity = 0.
    test_sentences = read_csv_data(test_file_path)
    lm = LanguageModel(model_path, document_root_path + 'char2index.pickle', document_root_path + 'index2char.pickle')
    with open(document_root_path + 'new_samples.txt', 'w', encoding='utf8') as news_sample_file:
        with open(document_root_path + 'probabilities.txt', 'w') as probabilities_file:
            for line, sentence in enumerate(test_sentences):
                hypothetical_sentence = lm.generate_new_sample(sentence.split()[0])

                # as we added '\\t' and '\\n' in sentence formation
                eval_sentence = lm.evaluation(sentence[2:-2])
                avg_cer += eval_sentence['CER']
                avg_perplexity += eval_sentence['Perplexity']
                news_sample_file.write(hypothetical_sentence + '\n')
                probabilities_file.write(str(lm.get_overall_probability(hypothetical_sentence)) + '\n')
                # break
                if line % 1000 == 0 and line != 0:
                    print('line:    {}  cer:    {}  perplexity:     {}'.format(line, avg_cer / line,
                                                                               avg_perplexity / line))
                    break


if __name__ == '__main__':
    print('Language Model')
    generate_sample_using_test_file(given_doc_root_path + 'test.csv', document_root_path + 'CharRNN_1.pth')

