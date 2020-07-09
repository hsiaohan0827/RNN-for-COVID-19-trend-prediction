import inspect
import math
import os
import csv

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import pygal
import pygal_maps_world.maps
from pygal.maps.world import COUNTRIES


''' Training '''
def train(e, model, optimizer, train_loader, batch_size):
    device = torch.device("cuda")
    model.train()
    
    # using BCE loss w/ sigmoid
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    train_loss = 0
    train_acc = 0
    for batch in train_loader:
        input, label = batch
        input = input.to(device).float()
        label = label.to(device).float()
        
        optimizer.zero_grad()
        model.zero_grad()
        
        # calculate loss & accuracy
        result= model(input)
        loss = criterion(result, label)
        acc = cal_accuracy(result, label)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += acc
        #print("[Epoch #{}] loss: {} acc: {}".format(e, loss, acc))

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    loss = {
        'BCE': train_loss,
        'accuracy': train_acc
    }
    return loss


''' Testing '''
def test(model, test_loader):
    model.eval()
    device = torch.device("cuda")
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    test_loss = 0
    test_acc = 0
    #t = tqdm(test_loader)
    #for batch in t:
    for batch in test_loader:
        input, label = batch
        input = input.to(device).float()
        label = label.to(device).float()
        
        result = model(input)
        loss = criterion(result, label)
        acc = cal_accuracy(result, label)

        test_loss += loss.item()
        test_acc += acc

    
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)
    loss = {
        'BCE': test_loss,
        'accuracy': test_acc
    }
    return loss

''' Plotting Country trend state by pygal'''
def predict(model, test_loader):
    model.eval()
    device = torch.device("cuda")

    # collecting countries name
    with open('covid_19.csv', newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        rows = rows[3:]
    country_list = [ country[0] for country in rows ]

    # input last seq of days to predict next state
    for batch in test_loader:
        input = batch
        input = input.to(device).float()
        
        result = model(input)
        m = nn.Sigmoid()
        pred = m(result).tolist()
        #print(pred)
    
    # making plot
    print('starting plotting world map')
    wm = pygal_maps_world.maps.World()
    wm.title = 'Covid-19 trend'

    country_code = []
    for country_name in range(len(country_list)):
        country_code.append(get_country_code(country_list[country_name]))
    
    inc = {}
    dec = {}
    for country in range(len(country_code)):
        if pred[country][0] >= 0.5:
            inc[country_code[country]] = pred[country][0]
        else:
            dec[country_code[country]] = pred[country][0]
    wm.add('Increasing', inc)
    wm.add('Decreasing', dec)
    wm.render_to_png('Covid-19 trend.png')
    print('finished!')

def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    return None

def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['rnn'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'rnn': model.state_dict(),
    }, ckpt_fpath)

def cal_accuracy(result, label):
    m = nn.Sigmoid()
    pred = (m(result) >= 0.5).float()
    acc = pred.eq(label).sum().float() / len(label)
    return acc

