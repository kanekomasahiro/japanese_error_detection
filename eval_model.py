import torch.nn as nn
import torch
from torch.autograd import Variable
import model


class Model(object):
    def __init__(self, model_type):
        if model_type == 'word':
            eval_model = 'model/ja_model_word'
        elif model_type == 'char':
            eval_model = 'model/ja_model_char'

        checkpoint = torch.load(eval_model, map_location=lambda storage, loc: storage)
        self.model_hp = checkpoint['hp']
        self.vocab = checkpoint['vocab']

        rnn = model.Model(self.model_hp, self.vocab, model_type)

        rnn.load_state_dict(checkpoint['rnn'])

        rnn.cpu()

        self.rnn = rnn
        self.rnn.eval()


    def get_vocab(self):
        return self.vocab


    def predict(self, char_input, word_input):
        pre = self.rnn(char_input, word_input)
        return pre

