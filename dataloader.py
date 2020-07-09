import random

import csv
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset

class CovidDataset(Dataset):
    def __init__(self, isTrain, interval, thres, input_dim):
        if isTrain:
            self.data = np.load('training_pair-{}.npy'.format(thres))
        else:
            self.data = np.load('testing_pair-{}.npy'.format(thres))

        self.L = interval
        self.data_row = len(self.data[0]) - self.L -2
        self.input = input_dim

    def __len__(self):
        return len(self.data) * (self.data_row)

    def __getitem__(self, idx):
        country = idx // self.data_row
        start_idx = idx % self.data_row
        daily_diff = self.data[country]
        #start_idx = random.randint(0, self.data_row)
        if daily_diff[start_idx+self.L+1] > daily_diff[start_idx+self.L]:
            label = 1
        else:
            label = 0
        if self.input == 1:
            return np.array([[daily_diff[day_idx]] for day_idx in range(start_idx, start_idx+self.L)]), np.array([label])
        else:
            return np.array([[day_idx, daily_diff[day_idx]] for day_idx in range(start_idx, start_idx+self.L)]), np.array([label])

class CovidPredictDataset(Dataset):
    def __init__(self, interval, input_dim):

        with open('covid_19.csv', newline='') as csvfile:
            rows = list(csv.reader(csvfile))
            rows = rows[3:]
        self.country_list = [ country[0] for country in rows ]

        self.data = np.load('all_country.npy')
        self.L = interval
        self.input = input_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        daily_diff = self.data[idx]
        if self.input == 1:
            return np.array([[daily_diff[day_idx]] for day_idx in range((-1) * self.L, len(daily_diff))])
        else:
            return np.array([[day_idx, daily_diff[day_idx]] for day_idx in range(len(daily_diff)-self.L, len(daily_diff))])




train_data = CovidDataset(True, 10, 0, 1)
train_loader = Data.DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)
count1 = 0
count0 = 0
for batch in train_loader:
    for i in batch[1]:
        if i[0] ==1:
            count1 += 1
        else:
            count0 += 1

print(count1)
print(count0)