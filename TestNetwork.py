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

if __name__ == "__main__":

    testset = SpeechDataset(root_dir='./data/test')
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('zdrowy','nowotwory')

    criterion = nn.CrossEntropyLoss()

    # Load model with the best score
    model = Net()

    model.load_state_dict(torch.load('model.pt'))

    test_loss = 0.0

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    model.eval()
    # iterate over test data
    for data, target in testloader:
        # forward pass
        # data = data.unsqueeze(0).type(torch.FloatTensor)
        print(data.size())
        data = data.float()
        output = model(data)
        # calculate batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(4):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))