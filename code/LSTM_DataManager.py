# -*- coding: utf-8 -*-

import random
import numpy as np


# Train, Test 데이터를 불러오는 것을 담당해주는 클래스
class DataManager:

    def __init__(self, path='../../data/original_wafer_data/', answer_file='aIndex.txt', num_test_data=None):
        # path : wafer_data가 저장된 디렉토리
        # answer_file : 정답 파일 ( name-label pair)
        # num_test_data 는 지정하지 않으면 전체 데이터의 1/5로 지정됨 ( in load_data_name_and_labels() method )
        self.path = path
        self.answer_file = answer_file
        self.num_test_data = 0  # will be initialized in load_data_adn_labels() method

        self.train_data_labels, self.test_data_labels = self.load_data_name_and_labels(num_test_data)
        self.train_data, self.train_labels = self.load_data(self.train_data_labels)

        random.seed()

    def load_file(self, file_name):
        try:
            f = open(self.path + file_name, 'r')
            return f
        except FileNotFoundError:
            print("ERROR: File %s%s not exists, terminating process..." % (self.path, file_name))
            exit(0)

    # answer_file의 값을 [wafer name, label] 꼴로 나누고,
    # train, test용으로 나누어 리턴한다
    def load_data_name_and_labels(self, num_test_data):
        labels = self.load_file(self.answer_file)
        labels = labels.readlines()
        labels = [label.strip().split() for label in labels]
        random.shuffle(labels)

        if num_test_data is not None:
            self.num_test_data = num_test_data
            if self.num_test_data >= len(labels):
                print("[-] number of test data is too big, doing resize...")
            self.num_test_data = len(labels) // 2
        else:
            self.num_test_data = len(labels) // 5

        return labels[:-self.num_test_data], labels[-self.num_test_data:]

    # [name, label]을 보고 해당하는 파일들에서 sensor 데이터를 찾아온다
    def load_data(self, name_labels):

        data_list = []
        label_list = []

        for l in name_labels:
            wafer_name = l[0] + '.txt'
            label = int(l[1])
            sensor_data = self.load_file(wafer_name).readlines()[1:]  # remove sensor name
            sensor_data = np.array([data.split()[1:] for data in sensor_data], dtype=np.float32)

            data_list.append(sensor_data)
            label_list.append(label)
            # FOR_LSTM (MAY NEED FIX)
            #label_list.append([label])

        # data_list = [ (Matrix( num timesteps x num sensors ), label), ... ]
        return data_list, label_list

    def get_train_data(self, batch_size, get_all=False):
        train_data = []
        train_labels = []

        if not get_all:
            idxs = random.sample(range(len(self.train_labels)), batch_size)
        else:   # return all train_data
            idxs = range(len(self.train_labels))

        for idx in idxs:
            train_data.append(self.train_data[idx])
            train_labels.append(self.train_labels[idx])

        return train_data, train_labels

    def get_test_data(self):
        return self.load_data(self.test_data_labels)


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