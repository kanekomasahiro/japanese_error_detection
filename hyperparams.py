
class Hyperparams:

    predict_unit = 'word'
    word_embedding = '../embeddings/glove.txt'
    predict = 'multi'
    
    batch_size = 128
    learning_rate = 1.0
    optimizer = 'adadelta'
    unk_freq = -1
    lr_decay = 1
    early_stop = 7
    
    layers = 1
    word_emb_size = 300
    char_emb_size = 100
    char_rnn_size = 100
    rnn_size = 300
    attention_size = 100
    rnn_output_size = 200
    output_size = 20
    loss_parameter = 0.09
    param_init = 0.1
    brnn = True
    emb_init = True

    maxlen = -1 # sentence length
    PAD = 0
    UNK = 1

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
