import sys
import torch
from math import ceil
from collections import Counter
from gensim.models import KeyedVectors
from vocab import Vocab

from hyperparams_japanese_ged import Hyperparams as hp

torch.manual_seed(hp.seed)

def changeNumToZero(word):
    try:
        float(word)
        return '0'
    except ValueError:
        return word


def make_emb(vocab):
    d = {}
    w2v = KeyedVectors.load_word2vec_format(hp.word_embedding, binary=False)
    for i in range(4, len(vocab)):
        word = vocab.getLabel(i)
        if word in w2v:
            d[i] = w2v[word]

    return d


def make_vocab(fpaths, char_maxlen=0):
    #We need tokenized data.
    chars = []
    words = []
    for fpath in fpaths:
        for line in open(fpath):
            if line != '\n':
                word = line.split("\t")[0]
                if hp.replace_digits:
                    word = changeNumToZero(word)
                char = [char for char in word]
                char_len = len(char)
                chars += char
                char_maxlen = char_maxlen if char_maxlen > char_len else char_len
                words += [word.lower()]

    char2cnt = Counter(chars)
    word2cnt = Counter(words)
    char_vocab = Vocab(hp, [hp.PAD_WORD, hp.UNK_WORD])
    word_vocab = Vocab(hp, [hp.PAD_WORD, hp.UNK_WORD])
    for char, freq in char2cnt.items():
        char_vocab.add(char, freq)
    for word, freq in word2cnt.items():
        word_vocab.add(word, freq)

    return char_vocab, word_vocab, char_maxlen


def make_pad(batch, char=False):
    pad_batch = []
    max_len = max([len(b) for b in batch])
    if not char:
        pad_batch = [torch.cat((batch[i], torch.LongTensor([hp.PAD] * (max_len-len(batch[i]))))) for i in range(len(batch))]
    else:
        pad_batch = [torch.cat((batch[i], torch.LongTensor([[hp.PAD] * batch[i].size(1)] * (max_len-len(batch[i]))))) for i in range(len(batch))]

    return torch.stack(pad_batch)

def make_examples_pad(batch):
    pad_batch = []
    max_len = max([len(b) for b in batch])
    pad_batch = [batch[i] + [hp.PAD_WORD for _ in range((max_len-len(batch[i])))] for i in range(len(batch))]

    return pad_batch


def make_data(input_file, vocab, char_maxlen, shuffle=True):
    char_input_list = []
    word_input_list = []
    tag_list = []
    ilist = []
    tlist = []
    examples = [] # 評価時に実際の出力結果のファイルを作成するために使う
    for line in open(input_file):
        if line == '\n':
            if hp.maxlen > 0 and len(ilist) > hp.maxlen:
                ilist = []
                tlist = []
                continue

            if len(ilist) == 0:
                ilist = []
                tlist = []
                continue

            # idx and pad
            char_sentence = [torch.LongTensor([vocab["char"].getIdx(char) for char in word]) for word in ilist]
            word_sentence = torch.LongTensor([vocab["word"].getIdx(word.lower()) for word in ilist])
            if hp.tag_num > 2:
                tag = torch.LongTensor([int(tag) for tag in tlist])
            else:
                tag = torch.LongTensor([1 if tag == 'i' or tag == '1' else 0 for tag in tlist])
            word_input_list += [word_sentence]
            tag_list += [tag]
            char_sentence = [torch.cat((char, torch.LongTensor([hp.PAD] * (char_maxlen-len(char))))) for char in char_sentence]
            char_input_list += [torch.stack(char_sentence)]
            examples.append(ilist)
            ilist = []
            tlist = []
        else:
            word, tag = line.split()
            if hp.replace_digits:
                word = changeNumToZero(word)
            ilist += [word]
            tlist += [tag.strip()]

    if shuffle:
        print('shuffling sentences')
        perm = torch.randperm(len(word_input_list))
        char_input_list = [char_input_list[idx] for idx in perm]
        word_input_list = [word_input_list[idx] for idx in perm]
        examples = [examples[idx] for idx in perm]
        tag_list = [tag_list[idx] for idx in perm]

    # Padding
    word_input_batch = [make_pad(word_input_list[hp.batch_size*i:hp.batch_size*(i+1)]).t() for i in range(ceil(len(word_input_list) / hp.batch_size))]
    examples = [make_examples_pad(examples[hp.batch_size*i:hp.batch_size*(i+1)]) for i in range(ceil(len(word_input_list) / hp.batch_size))]
    tag_batch = [make_pad(tag_list[hp.batch_size*i:hp.batch_size*(i+1)]).t() for i in range(ceil(len(tag_list) / hp.batch_size))]
    tag_batch = [make_pad(tag_list[hp.batch_size*i:hp.batch_size*(i+1)]).t() for i in range(ceil(len(tag_list) / hp.batch_size))]
    char_input_batch = [make_pad(char_input_list[hp.batch_size*i:hp.batch_size*(i+1)], char=True) for i in range(ceil(len(char_input_list) / hp.batch_size))]

    return char_input_batch, word_input_batch, tag_batch, examples


def main():
    print("Building vocabulary ...")
    vocab = {}
    if hp.vocab_include_devtest:
        vocab["char"], vocab["word"], char_maxlen = make_vocab([hp.train_file, hp.valid_file])
    else:
        _, _, char_maxlen = make_vocab([hp.valid_file])
        vocab["char"], vocab["word"], char_maxlen = make_vocab([hp.train_file], char_maxlen=char_maxlen)
    vocab["word_pre_emb"] = make_emb(vocab["word"])
    print("Making train data ...")
    train = {}
    train["char"], train["word"], train["tag"], _ = make_data(hp.train_file, vocab, char_maxlen)
    print("Making validation data ...")
    valid = {}
    valid["char"], valid["word"], valid["tag"], _ = make_data(hp.valid_file, vocab, char_maxlen, shuffle=False)
    print("Saving data ...")
    save_dicts = {"vocab": vocab,
                  "train": train,
                  "valid": valid}
    torch.save(save_dicts, hp.save_data)


if __name__ == '__main__':
    main()
