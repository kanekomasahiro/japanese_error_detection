from eval_model import Model
import torch
import math
from torch.autograd import Variable
import sys
import MeCab


def main():
    args = sys.argv
    tagger = MeCab.Tagger("-Owakati")
    softmax = torch.nn.Softmax(dim=2)

    input = args[1]
    input = tagger.parse(input).strip()
    input = input.split()

    rnn = Model()

    char_maxlen = 15
    test = {}
    vocab = rnn.get_vocab()
    char_input = [torch.LongTensor([vocab["char"].getIdx(char) for char in word]) for word in input]
    char_input = torch.stack([torch.cat((char, torch.LongTensor([0] * (char_maxlen-len(char))))) for char in char_input])
    word_input = torch.LongTensor([vocab["word"].getIdx(word.lower()) for word in input])

    char_input = Variable(char_input.unsqueeze(0))
    word_input = Variable(word_input.view(-1, 1))
    word_pres = rnn.predict(char_input, word_input)
    word_pres = softmax(word_pres)[:,:,1].view(-1).tolist()

    print(input)
    print(word_pres)


if __name__ == "__main__":
    main()
