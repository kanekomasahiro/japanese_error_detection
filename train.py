import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torchcrf import CRF
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from math import ceil
import optim
import model
from hyperparams_japanese_ged import Hyperparams as hp

if hp.gpu:
    cuda.set_device(hp.gpu)
torch.manual_seed(hp.seed)

def eval(pres, tags, pros=None, label=1):
    precision, recall, fscore, _ =  precision_recall_fscore_support(
                                            tags, pres, average='binary')

    return precision, recall, fscore


def dev(rnn, criterion, data_char_input, data_word_input, data_tag, crf):
    total_loss = 0
    softmax = nn.Softmax(dim=1)

    rnn.eval()
    word_pres_list = []
    word_pros_list = []
    word_tags_list = []
    sen_pres_list = []
    sen_tags_list = []
    for i in range(len(data_word_input)):
        char_input_batch = data_char_input[i]
        word_input_batch = data_word_input[i]
        tag_batch = data_tag[i]
        char_input_batch = Variable(char_input_batch)
        word_input_batch = Variable(word_input_batch)
        word_tag = Variable(tag_batch)
        if hp.gpu >= 0:
            char_input_batch = char_input_batch.cuda()
            word_input_batch = word_input_batch.cuda()
            word_tag = word_tag.cuda()
        sen_tag = (torch.sum(word_tag, 0) != 0).float()
        word_pres = rnn(char_input_batch, word_input_batch)
        '''
        if hp.crf:
            loss = -crf(word_pres, word_tag.long())
            total_loss += loss.item()
            word_tag = word_tag.view(-1)
            word_tags_list += [word_tag.data]
            word_pres_list += [torch.LongTensor(crf.decode(word_pres)).view(-1)]
            word_pros_list += [word_pres.data.view(-1)]
        else:
        '''
        word_pres = word_pres.view(-1, word_pres.size(-1))
        word_tag = word_tag.view(-1)
        loss = criterion(word_pres, word_tag)
        total_loss += loss.item()
        word_tags_list += [word_tag.data]
        word_pres_list += [torch.argmax(softmax(word_pres), 1).data.view(-1)]
        word_pros_list += [word_pres.data.view(-1)]
    word_pres = torch.cat(word_pres_list).cpu().numpy()
    word_pros = torch.cat(word_pros_list).cpu().numpy()
    word_tags = torch.cat(word_tags_list).cpu().numpy()

    word_precision, word_recall, word_fscore =\
                                eval(word_pres, word_tags, word_pros, label=i)

    return total_loss, word_precision, word_recall, word_fscore


def trainModel(rnn, rnn_optim, dataset, vocab):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    #crf = CRF(hp.tag_num)
    #crf = crf.cuda()
    #crf_optim = optim.Optim(hp.optimizer, hp.learning_rate, hp.lr_decay)
    #crf_optim.set_parameters(crf.parameters())

    def forward():
        rnn.train()
        sen_pres_list = []
        sen_tags_list = []
        word_pros_list = []
        word_pres_list = []
        word_tags_list = []
        batchOrder = torch.randperm(len(dataset['train']['word']))
        total_loss, num_correct, total_tgt_words = 0, 0, 0
        num_src_correct, total_src_words = 0, 0
        num_tagCorrect = 0
        start = time.time()
        for i in range(len(dataset['train']['word'])):
            rnn.zero_grad()
            #crf.zero_grad()
            batchIdx = batchOrder[i]
            char_input_batch = Variable(dataset['train']['char'][batchIdx])
            word_input_batch = Variable(dataset['train']['word'][batchIdx])
            word_tag = Variable(dataset['train']['tag'][batchIdx])
            if hp.gpu >= 0:
                char_input_batch = char_input_batch.cuda()
                word_input_batch = word_input_batch.cuda()
                word_tag = word_tag.cuda()
            word_pres = rnn(char_input_batch, word_input_batch)
            '''
            if hp.crf:
                loss = -crf(word_pres, word_tag.long())
                loss.backward()
                rnn_optim.step()
                crf_optim.step()
                total_loss += loss.item()
                word_tag = word_tag.view(-1)
                word_tags_list += [word_tag.data]
                word_pres_list += [torch.LongTensor(crf.decode(word_pres)).view(-1)]
                word_pros_list += [word_pres.data.view(-1)]
            else:
            '''
            word_pres = word_pres.view(-1, word_pres.size(-1))
            word_tag = word_tag.view(-1)
            loss = criterion(word_pres, word_tag)
            loss.backward()
            rnn_optim.step()
            total_loss += loss.item()
            word_tags_list += [word_tag.data]
            word_pres_list += [torch.argmax(softmax(word_pres), 1).data.view(-1)]
            word_pros_list += [word_pres.data.view(-1)]

        word_pres = torch.cat(word_pres_list).cpu().numpy()
        word_pros = torch.cat(word_pros_list).cpu().numpy()
        word_tags = torch.cat(word_tags_list).cpu().numpy()

        word_precision, word_recall, word_fscore = eval(word_pres, word_tags, word_pros, label=i)

        return total_loss, word_precision, word_recall, word_fscore

    print('Start training')
    max_fscore = -1
    for epoch in range(1, hp.epochs + 1):
        print("epoch{}:".format(epoch))
        train_loss, word_precision, word_recall, word_fscore = forward()
        print('train loss: {:.2f} word_precision:{:.2f} word_recall:{:.2f} word_fscore:{:.2f}'.format(
                                                train_loss, word_precision*100, word_recall*100, word_fscore*100))
        valid_loss, word_precision, word_recall, word_fscore = dev(
                                                rnn, criterion, dataset['valid']['char'], dataset['valid']['word'], dataset['valid']['tag'], None)
        print('valid loss: {:.2f} word_precision:{:.2f} word_recall:{:.2f} word_fscore:{:.2f}'.format(valid_loss, word_precision*100, word_recall*100, word_fscore*100))
        if word_fscore > max_fscore:
            max_epoch = epoch
            max_fscore = word_fscore
            rnn_state_dict = rnn.state_dict()
            checkpoint = {
                'rnn': rnn_state_dict,
                'vocab': vocab,
                'hp': hp,
                'rnn_optim': rnn_optim
            }
            torch.save(checkpoint, '{}_checkpoint'.format(hp.save_model))
        if epoch - max_epoch >= hp.early_stop:
            break
    checkpoint = torch.load('{}_checkpoint'.format(hp.save_model))
    torch.save(checkpoint,
               '{}_acc_{:.2f}_epoch_{}.pt'.format(hp.save_model, max_fscore*100, max_epoch))
    import os
    os.remove('{}_checkpoint'.format(hp.save_model))

def main():

    print("Loading data from...")
    dataset = torch.load(hp.save_data)
    vocab = {}
    vocab['word'] = dataset['vocab']['word']
    vocab['char'] = dataset['vocab']['char']

    print('Building model...')
    rnn = model.Model(hp, vocab)

    if hp.param_init:
        print('Intializing model parameters.')
        for p in rnn.parameters():
            p.data.uniform_(-hp.param_init, hp.param_init)

    if hp.emb_init:
        print('Intializing embeddings.')
        w2v = dataset['vocab']['word_pre_emb']
        for i in range(rnn.word_lut.weight.size(0)):
            if i in w2v:
                rnn.word_lut.weight[i].data.copy_(torch.from_numpy(w2v[i]))

    if hp.gpu >= 0:
        rnn.cuda()
    else:
        rnn.cpu()

    rnn_optim = optim.Optim(hp.optimizer, hp.learning_rate, hp.lr_decay)
    rnn_optim.set_parameters(rnn.parameters())

    trainModel(rnn, rnn_optim, dataset, vocab)


if __name__ == "__main__":
    main()
