import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from math import ceil


class char_RNN(nn.Module):

    def __init__(self, hp, vocab):
        layers = hp.layers
        num_directions = 2 if hp.brnn else 1
        char_vocab_size = len(vocab['char'])

        super(char_RNN, self).__init__()
        self.char_lut = nn.Embedding(char_vocab_size,
                                    hp.char_emb_size,
                                    padding_idx=hp.PAD)
        self.rnn = nn.LSTM(hp.char_emb_size, hp.char_rnn_size,
                        num_layers=layers,
                        bidirectional=hp.brnn)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        input = input.view(-1, input.size(-1))
        input = input.t()
        emb = self.char_lut(input)
        outputs, hiddens = self.rnn(emb, hidden)
        char_emb = torch.cat((hiddens[0][0], hiddens[0][1]), 1)
        char_emb = char_emb.view(-1, batch_size, char_emb.size(-1))
        return char_emb


class Model(nn.Module):

    def __init__(self, hp, vocab):
        self.model_type = hp.model_type # word or char&word
        layers = hp.layers
        num_directions = 2 if hp.brnn else 1
        assert hp.rnn_size % num_directions == 0
        word_emb_size = hp.word_emb_size
        char_rnn_size = hp.char_rnn_size
        word_vocab_size = len(vocab['word'])
        self.tag_num = hp.tag_num

        super(Model, self).__init__()
        self.word_lut = nn.Embedding(word_vocab_size,
                                  word_emb_size,
                                  padding_idx=hp.PAD)
        self.dropout = nn.Dropout(p=hp.dropout_rate)

        if self.model_type == 'word':
            self.rnn = nn.LSTM(word_emb_size, hp.rnn_size,
                            num_layers=layers,
                            bidirectional=hp.brnn)
        if self.model_type == 'char&word':
            self.char_lut = char_RNN(hp, vocab)
            self.rnn = nn.LSTM(word_emb_size+char_rnn_size*2, hp.rnn_size,
                            num_layers=layers,
                            bidirectional=hp.brnn)

        self.linear_rnn = nn.Linear(hp.rnn_size*2, hp.rnn_output_size)
        self.tanh = nn.Tanh()
        self.linear_pre = nn.Linear(hp.rnn_output_size, self.tag_num)
        #self.linear_in = nn.Linear(hp.rnn_output_size, hp.attention_size)
        #self.linear_pre = nn.Linear(hp.attention_size, self.tag_num)

    def forward(self, char_input, word_input, hidden=None):
        word_emb = self.word_lut(word_input)
        if self.model_type == 'word':
            emb = self.dropout(word_emb)
        elif self.model_type == 'char&word':
            char_emb = self.char_lut(char_input)
            emb = self.dropout(torch.cat((word_emb, char_emb), -1))
        outputs, _ = self.rnn(emb)
        outputs = self.dropout(outputs)
        hiddens = self.tanh(self.linear_rnn(outputs))
        #d = self.tanh(self.linear_in(hiddens))
        word_pres = self.linear_pre(hiddens)

        return word_pres
