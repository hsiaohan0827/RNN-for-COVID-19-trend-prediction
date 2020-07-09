import os
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from dataloader import CovidDataset, CovidPredictDataset
from model import RNNModel
from utils import train, test, predict, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs.config", help="module path of the configuration file.")
    return parser.parse_args()

def build_model(C):
    decoder = RNNModel(C.rnn.model_type, C.rnn.input_dim, C.rnn.hidden_dim, C.rnn.layer_dim, C.rnn.dropout)
    print(decoder)
    return decoder

def log_train(C, summary_writer, e, loss, lr):
    summary_writer.add_scalar(C.tx_train_loss, loss['BCE'], e)
    summary_writer.add_scalar(C.tx_train_acc, loss['accuracy'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    print("[Epoch #{}] train loss: {} train accuracy: {}".format(e, loss['BCE'], loss['accuracy']))

def log_test(C, summary_writer, e, loss):
    summary_writer.add_scalar(C.tx_val_loss, loss['BCE'], e)
    summary_writer.add_scalar(C.tx_val_acc, loss['accuracy'], e)
    print("[Epoch #{}] test loss: {} test accuracy: {}".format(e, loss['BCE'], loss['accuracy']))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':

    # set config
    args = parse_args()
    C = importlib.import_module(args.config).TrainConfig
    print("MODEL ID: {}".format(C.model_id))

    # create dataloader
    train_data = CovidDataset(True, C.interval, C.thres, C.rnn.input_dim)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=C.batch_size, shuffle=True)
    #train_loader = Data.DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=True)

    test_data = CovidDataset(False, C.interval, C.thres, C.rnn.input_dim)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=C.batch_size, shuffle=False)
    #test_loader = Data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)

    print('train data: '+str(len(train_data)))
    print('test data: '+str(len(test_data)))

    # logs
    summary_writer = SummaryWriter(C.log_dpath)

    if C.isTrain:

        # model
        model = build_model(C)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=C.lr)
        #lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                     #patience=C.lr_decay_patience, verbose=True)

        device = torch.device("cuda")
        model = model.to(device)

        for e in range(1, C.epochs + 1):
            ckpt_fpath = C.ckpt_fpath_tpl.format(e)

            """ Train """
            print("\n")
            train_loss = train(e, model, optimizer, train_loader, C.batch_size)
            log_train(C, summary_writer, e, train_loss, get_lr(optimizer))

            """ Test """
            test_loss = test(model, test_loader)
            log_test(C, summary_writer, e, test_loss)

            if e >= C.save_from and e % C.save_every == 0:
                print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                save_checkpoint(e, model, ckpt_fpath, C)

            #if e >= C.lr_decay_start_from:
                #lr_scheduler.step(test_loss['BCE'])
    else:
        # load model
        model = build_model(C)
        model = load_checkpoint(model, C.ckpt_fpath)
        
        device = torch.device("cuda")
        model = model.to(device)

        test_data = CovidPredictDataset(C.interval, C.rnn.input_dim)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
        
        predict(model, test_loader)
