# -*- coding: utf-8 -*-

import random
import numpy as np


# Database class for managing train and test data
class Database:

    def __init__(self, train_path='../../data/', train_answer_file='trainIndex.txt',
                test_path='../../data/', test_answer_file='testIndex.txt', sufix=''):
        # train_path, test_path : directory for train, test data
        # train_answer_file, test_answer_file : name-label pair for train, test data
        # sufix : sufix for choosing specific file name
        random.seed()
        self.train_path = train_path
        self.train_answer_file = train_answer_file
        self.test_path = test_path
        self.test_answer_file = test_answer_file

        self.train_data_labels = self.load_train_data_and_labels()
        self.train_data, self.train_labels = self.load_data(self.train_data_labels, type='train', sufix=sufix)

        self.test_data_labels = self.load_test_data_and_labels()
        self.test_data, self.test_labels = self.load_data(self.test_data_labels, type='test', sufix=sufix)

    def load_file(self, file_name, type):
        try:
            path = None
            if type == 'train':
                path = self.train_path
            elif type == 'test':
                path = self.test_path
            else:
                print("ERROR: type must be 'train' or 'test'")
                exit(0)

            f = open(path + file_name, 'r')
            return f

        except FileNotFoundError:
            print("ERROR: File %s%s not exists, terminating process..." % (path, file_name))
            exit(0)

    # get (wafer name, label) from train answer file
    def load_train_data_and_labels(self):
        labels = self.load_file(self.train_answer_file, type='train')
        labels = labels.readlines()
        labels = [label.strip().split() for label in labels]
        random.shuffle(labels)

        return labels

    # get (wafer name, label) from test answer file
    def load_test_data_and_labels(self):

        labels = self.load_file(self.test_answer_file, type='test')
        labels = labels.readlines()
        labels = [label.strip().split() for label in labels]
        random.shuffle(labels)

        return labels

    # get sensor data from matching file name by checking (name, label)
    def load_data(self, name_labels, type, sufix=''):

        data_list = []
        label_list = []

        for n, l in name_labels:
            wafer_name = n + sufix + '.txt'
            label = int(l)
            sensor_data = self.load_file(wafer_name, type=type).readlines()[1:]  # remove sensor name
            sensor_data = np.array([data.split()[1:] for data in sensor_data], dtype=np.float32)  # remove time

            data_list.append(sensor_data)
            #label_list.append(label)
            # FOR_LSTM (MAY NEED FIX)
            label_list.append([label])

        # data_list = [ (Matrix( num timesteps x num sensors ), label), ... ]
        return data_list, label_list

    def get_train_data(self, batch_size=10, get_all=False):
        train_data = []
        train_labels = []

        # return all train data in database
        if get_all:
            idxs = range(len(self.train_labels))
        else:
            idxs = random.sample(range(len(self.train_labels)), batch_size)

        for idx in idxs:
            train_data.append(self.train_data[idx])
            train_labels.append(self.train_labels[idx])

        return train_data, train_labels

    def get_test_data(self):
        return self.test_data, self.test_labels


'''
# for debug

print('[*] Loading DataManager')
d = DataManager()
print('[*] Load Done')

for i in range(10):
    data, labels = d.get_train_data(10)

    for j in range(len(labels)):
        print('---------train-----------')
        print(data[j][0], labels[j])

    data, labels = d.get_test_data()

    for j in range(len(labels)):
        print('---------test-----------')
        print(data[j][0], labels[j])
'''
