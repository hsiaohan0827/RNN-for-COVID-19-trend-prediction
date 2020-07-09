import os
import time


class RNNConfig:
    model_type = 'LSTM' 
    input_dim = 1
    hidden_dim = 32
    layer_dim = 1
    dropout = 0

class TrainConfig:

    isTrain = False
    interval = 10
    thres = 0

    """ Model """
    rnn = RNNConfig

    """ Optimization """
    epochs = 500
    batch_size = 500
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 1e-5
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    lr_decay_start_from = 0


    """ Pretrained Model """
    pretrained_path = ''

    """ ID """
    exp_id = rnn.model_type
    L_id = 'L-{}'.format(interval)
    batch_id = 'bs-{}'.format(batch_size)
    optim_id = optimizer + ' lr-{}'.format(lr)
    layer_id = 'layer-{}'.format(rnn.layer_dim)
    input_id = 'in-{}'.format(rnn.input_dim)
    hidden_id = 'hid-{}'.format(rnn.hidden_dim)
    dropout_id = 'dr-{}'.format(rnn.dropout)
    thres_id = 'th-{}'.format(thres)
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = "|".join([ exp_id, L_id, batch_id, input_id, layer_id, hidden_id, thres_id, optim_id ])

    """ Log """
    log_dpath = "half_logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 100

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_acc = "accuracy/train"
    tx_val_loss = "loss/test"
    tx_val_acc = "accuracy/test"

    tx_lr = "params/lr"

    """ predict """
    ckpt = 'LSTM|L-10|bs-500|in-1|layer-1|hid-32|th-0|Adam lr-0.001'
    epoch = 500
    
    ckpt_fpath = os.path.join("checkpoints", ckpt)
    ckpt_fpath = os.path.join(ckpt_fpath, str(epoch)+'.ckpt')