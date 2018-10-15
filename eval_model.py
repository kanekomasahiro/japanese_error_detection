import torch.nn as nn
import torch
from torch.autograd import Variable
import model


class Model(object):
    def __init__(self):
        eval_model = 'model/ja_model'

        checkpoint = torch.load(eval_model, map_location=lambda storage, loc: storage)
        self.model_hp = checkpoint['hp']
        self.vocab = checkpoint['vocab']
        self.pre = self.model_hp.predict

        rnn = model.Model(self.model_hp, self.vocab)

        rnn.load_state_dict(checkpoint['rnn'])

        rnn.cpu()

        self.rnn = rnn
        self.rnn.eval()


    def get_vocab(self):
        return self.vocab


    def predict(self, char_input, word_input):
        pre = self.rnn(char_input, word_input)
        return pre

