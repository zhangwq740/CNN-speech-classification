import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from scipy import signal
import torchvision
import os
from glob import glob
import numpy as np
from sklearn.preprocessing import normalize
from os import path
import csv
import torch
import scipy.io
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SpeechDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
        :param csv_file: Path to the csv file
        :param root_dir: Directory with all data
        :param transform: Optional transform
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        csv_files = glob(os.path.join(self.root_dir, '*.csv'))
        csv_files = [x.split(sep='\\')[1] for x in csv_files]

        return len(csv_files)

    def __getitem__(self, idx):
        csv_files = glob(os.path.join(self.root_dir, '*.csv'))
        csv_files = [x.split(sep='\\')[1] for x in csv_files]

        sample = {}

        for index, file in enumerate(csv_files):
            filepath = os.path.join(self.root_dir, file)
            sample.clear()
            if idx == index:
                # load stft to numpy
                stft = np.loadtxt(open(filepath, "rb"), delimiter=" ", skiprows=0)
                # print(stft.shape)
                if stft.shape[1] != 126:
                    continue
                # take name of file
                if np.any(np.isnan(stft)):
                    continue
                name = file.split(' ')
                # self.plot_stft(stft, name[0])
                tran1 = transforms.ToTensor()
                stft = tran1(stft)
                # assign class indices
                class_indices = {'zdrowy' : 0 , 'nowotwory' : 1}
                # return tensor and label 0 or 1
                return stft, class_indices[name[0]]

    def plot_stft(self, stft, title):
        plt.figure()
        plt.imshow(stft)
        plt.title(title)
        plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(15360, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # print("ROZMIAR W SIECI PO KOLEI")
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        size = x.size()
        x = x.view(-1, size[1]*size[2]*size[3])
        # add dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.size())
        return x

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

if __name__ == "__main__":

    trainset = SpeechDataset(root_dir='./data/trening')
    testset = SpeechDataset(root_dir='./data/test')

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('zdrowy','nowotwory')

    # model
    model = Net()
    print(model)

    # specify loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # # TRAINING
    n_epochs = 50

    train_loss_min = np.Inf
    total_correct = 0

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in trainloader:
            # clear the gradients
            #print(data.size())
            #data = data.unsqueeze(0).type(torch.FloatTensor)
            optimizer.zero_grad()
            # forward pass
            data = data.float()
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            # update training loss
            # print("Loss.item():", loss.item())
            # print("Data.size(0):", data.size())
            train_loss += loss.item()
            total_correct += get_num_correct(output, target)
            # print("Train loss: ", train_loss)

        # calculate average loss
        train_loss = train_loss/len(trainloader.dataset)

        # print training/validation stats
        print('Epoch: {} \tTraining Loss: {:.6f} Correct guesses: {:.6f}'.format(epoch, train_loss, total_correct))

        # save model if validation loss has decreased
        if train_loss <= train_loss_min:
            print('Validation loss decreased from {:.6f} --> {:.6f}. Saving model...'.format(
                train_loss_min,
                train_loss))
            torch.save(model.state_dict(), 'model.pt')
            train_loss_min = train_loss




