
class Hyperparams:
    train_file = '../data/arai_ja_ged_data/train.tsv'
    valid_file = '../data/arai_ja_ged_data/dev.tsv'
    test_file = '../data/arai_ja_ged_data/test.tsv'

    save_data = 'model_data/ja_data'
    save_model = 'model/ja_model'

    replace_digits = True
    vocab_include_devtest = False
    no_char = True

    tag_num = 2
    word_embedding = '../embeddings/glove.txt'
    
    batch_size = 64
    learning_rate = 1.0
    optimizer = 'adadelta'
    unk_freq = 1
    lr_decay = 1
    early_stop = 7
    
    layers = 1
    word_emb_size = 300
    char_emb_size = 100
    char_rnn_size = 100
    rnn_size = 300
    attention_size = 100
    rnn_output_size = 200
    output_size = 200
    param_init = 0.1
    brnn = True
    emb_init = True

    maxlen = -1 # sentence length
    epochs = 20
    dropout_rate = 0.
    seed = 100
    
    PAD = 0
    UNK = 1

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'

    gpu = 1
