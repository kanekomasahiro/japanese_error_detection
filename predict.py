from eval_model import Model
import torch
import math
from torch.autograd import Variable
import MeCab


def detect_errors(input, threshold=0.5):
    tagger = MeCab.Tagger("-Owakati")
    softmax = torch.nn.Softmax(dim=2)

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
    scores = rnn.predict(char_input, word_input)
    scores = softmax(scores)[:,:,1].view(-1).tolist()
    labels = [int(score > threshold) for score in scores]

    return input, labels


if __name__ == "__main__":
    import sys
    args = sys.argv
    input, scores = detect_errors(args[1])
    print(input)
    print(scores)
