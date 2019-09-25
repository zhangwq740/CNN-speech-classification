import csv
import wave
from scipy.io import wavfile
import numpy as np
import os
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt
from glob import glob
import pyaudio
import wave
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from keras.utils import to_categorical

# open folders -> take files .wav
# wav -> spectrograms
# save spectrograms in folders
# load data to train/validate/test

FS = 44100

class DataLoader():

    def generate_data(self):
        DIR = 'data'
        wav_files = glob(os.path.join(DIR, '*/*wav'))
        wav_files = [x.split(sep='\\')[1] + '/' + x.split(sep='\\')[2] for x in wav_files]
        data = []

        for e in wav_files:
            label, name = e.split('/')
            file = os.path.join(DIR, e)
            f, t, Zxx = self.wave_to_spec(file)
            # Jesli jest dluzy od 100 ramek
            # if Zxx.shape[1] > 100:
            print("[", Zxx.shape[0], ' ,', Zxx.shape[1], "]")
            filepath = os.path.join(DIR, label, 'data_csv')
            self.save_spec_to_csv(Zxx, filepath, name)
            # self.plot_spectrogram(f, t, Zxx, file)
            # else:
            #    continue

    # normalization
    def normalize(self, data):
        data = data / np.max(data)
        return data

    # audio file that's shorter than 44100 samples is filled with 0's
    def pad_audio(self, y, fs):
        if len(y) >= fs: return y
        else: return np.pad(y, pad_width=(fs - len(y), 0), mode='constant', constant_values=(0,0))

    # audio file that is longer than 44100 is randomly cut to 1s duration
    def chop_audio(self, y, fs):
        if len(y) <= fs: return y
        else:
            beginSample = np.random.randint(0, len(y) - fs)
            return y[beginSample : (beginSample + fs)]

    # preemphasis filtering
    def preemphasis_filtering(self, data, pre_emphasis=0.97):
        pre_emphasis = 0.97
        data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
        return data

    def read_wave(self, wav_name):
        obj = wave.open(wav_name, 'r')
        num_of_channels = obj.getnchannels()
        if num_of_channels == 1:
            fs, y = wavfile.read(wav_name)
            return fs, y

    # fs=44100
    def wave_to_spec(self, wav_name, bLog_spec=True, threshold_freq_down = None, threshold_freq_up = None):
        fs, y = self.read_wave(wav_name)
        # resampling
        if fs != 16000:
            # (signal, amount of samples in resampled signal)
            print(fs)
            fs_new = 16000
            y = signal.resample(y, int((fs_new/fs) * y.shape[0]))
            print('Nowa długość sygnału: ', len(y))
            fs = fs_new
        y = self.pad_audio(y, fs)
        y = self.chop_audio(y, fs)
        y = self.normalize(y)
        y = self.preemphasis_filtering(y)
        y = self.change_zero_to_something_small(y)
        # STFT
        f, t, Zxx = stft(y, fs, window='hann')
        # LOW PASS
        if threshold_freq_up is not None:
            Zxx = Zxx[f <= threshold_freq_up, :]
            f = f[f <= threshold_freq_up]
        # HIGH PASS
        if threshold_freq_down is not None:
            Zxx = Zxx[f >= threshold_freq_down, :]
            f = f[f >= threshold_freq_down]
            # Logarithm of spectrogram
        if bLog_spec:
            Zxx_log = np.log(np.abs(Zxx))
            return f, t, Zxx_log
        else:
            return f, t, Zxx

    def plot_spectrogram(self, f, t, Zxx, name):
        plt.pcolormesh(t, f, Zxx)
        plt.title(name)
        plt.show()

    def change_zero_to_something_small(self, x):
        for i in range(len(x)):
            if x[i] == 0:
                x[i] = random.uniform(0.0000001, 0.0000002)
        return x

    def save_spec_to_csv(self, Zxx, filepath, filename):
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
            print("Successfully created the directory %s" % filepath)

        # CSV
        with open((filepath + '\\' + filename + '.csv'), "w+", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ')
            csv_writer.writerows(Zxx)

dl = DataLoader()
dl.generate_data()