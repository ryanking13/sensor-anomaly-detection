# -*- coding: utf-8 -*-

import random
import numpy as np


# Train, Test 데이터를 불러오는 것을 담당해주는 클래스
class DataManager:

    def __init__(self, train_path='../../data/sliced/', train_answer_file='bIndex.txt',
                test_path='../../data/reduced/', test_answer_file='aIndex.txt',
                num_test_data=None):
        # path : wafer_data가 저장된 디렉토리
        # answer_file : 정답 파일 ( name-label pair)
        # num_test_data 는 지정하지 않으면 전체 데이터의 1/5로 지정됨 ( in load_data_name_and_labels() method )

        random.seed()
        self.train_path = train_path
        self.train_answer_file = train_answer_file
        self.test_path = test_path
        self.test_answer_file = test_answer_file
        self.num_test_data = num_test_data

        self.train_data_labels = self.load_train_data_and_labels()
        self.train_data, self.train_labels = self.load_data(self.train_data_labels, type='train')

        self.test_data_labels = self.load_test_data_and_labels()

    def load_file(self, file_name, type):
        try:
            path = None
            if type == 'train':
                path = self.train_path
            elif type == 'test':
                path = self.test_path
            else:
                print('Illegal type')
                exit(0)

            f = open(path + file_name, 'r')
            return f
        except FileNotFoundError:
            print("ERROR: File %s%s not exists, terminating process..." % (self.path, file_name))
            exit(0)

    # answer_file의 값을 [wafer name, label] 꼴로 나누어 리턴한다
    def load_train_data_and_labels(self):
        labels = self.load_file(self.train_answer_file, type='train')
        labels = labels.readlines()
        labels = [label.strip().split() for label in labels]
        random.shuffle(labels)

        return labels

    # answer_file의 값을 [wafer name, label] 꼴로 나누어 리턴한다
    def load_test_data_and_labels(self):

        labels = self.load_file(self.test_answer_file, type='test')
        labels = labels.readlines()
        labels = [label.strip().split() for label in labels]
        random.shuffle(labels)

        return labels[:self.num_test_data]


    # [name, label]을 보고 해당하는 파일들에서 sensor 데이터를 찾아온다
    def load_data(self, name_labels, type):

        data_list = []
        label_list = []

        for l in name_labels:
            wafer_name = l[0] + '.txt'
            label = int(l[1])
            sensor_data = self.load_file(wafer_name, type=type).readlines()[1:]  # remove sensor name
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
        return self.load_data(self.test_data_labels, type='test')


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
