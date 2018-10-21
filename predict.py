from eval_model import Model
import torch
import math
from torch.autograd import Variable
import MeCab


def detect_errors(input, model_type='word', threshold=0.5):
    softmax = torch.nn.Softmax(dim=2)
    if model_type == 'word':
        tagger = MeCab.Tagger("-Owakati")

        input = tagger.parse(input).strip()
        input = input.split()
    elif model_type == 'char':
        input = list(input)

    rnn = Model(model_type)

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

    return input, labels, scores


if __name__ == "__main__":
    import sys
    args = sys.argv
    input, labels, scores = detect_errors(args[1], model_type=args[2])
    print(input)
    print(labels)
    print(scores)
