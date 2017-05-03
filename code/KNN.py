# -*- coding: utf-8 -*-

import numpy as np

class KNN:

    def __init__(self, k=1):
        self.k = k  # num of neighbors
        self.train_data_set = []    # [ Matrix1( num of sensors * timesteps ), ... ]
        self.train_label_set = []   # [ wafer label1, ... ]

    # 기존에 학습한 Train data를 로드한다.
    # TODO: implement
    def load_train_data(self):
        return

    # Train data를 반환한다.
    def get_train_data(self):
        return self.train_data_set, self.train_label_set

    # 새로운 train data를 추가한다.
    def add_train_data(self, train_data, train_label):
        self.train_data_set.append(train_data)
        self.train_label_set.append(train_label)

    # 두 Matrix의 uclidean distance를 구한다.
    # 두 Matrix의 크기가 같음을 가정
    # 다를 경우 수정이 필요함
    def get_distance(self, mat1, mat2):
        dist = np.sum(np.absolute(mat1-mat2))
        return dist

    # 각 Train data에 대해 Test data와의 거리를 구한다
    # 가장 가까운 k개의 data(neighbor)의 label을 반환한다
    def get_neighbors(self, train_data_set, train_label_set, test_data):
        distances = []
        
        for i in range(len(train_data_set)):
            dist = self.get_distance(np.array(train_data_set[i]), np.array(test_data))
            distances.append((dist, train_label_set[i]))

        distances.sort()
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][1])

        return np.array(neighbors)

    # Test data(웨이퍼)의 양불을 추정하고 정확도를 확인한다.
    def test(self, test_data_set, test_label_set):
        predicted = self.predict(test_data_set)
        test_label_set = np.array(test_label_set)
        accuracy = 1 - (predicted != test_label_set).sum()/float(predicted.size)
        return test_label_set, predicted, accuracy

    # Test data(웨이퍼)의 양불을 추정한다.
    def predict(self, test_data_set):
        train_data_set, train_label_set = self.get_train_data()

        predicted = []
        for test_data in test_data_set:
            neighbors = self.get_neighbors(train_data_set, train_label_set, test_data)
            predicted.append(np.bincount(neighbors).argmax())

        return np.array(predicted)
