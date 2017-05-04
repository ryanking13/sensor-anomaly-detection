# -*- coding: utf-8 -*-

import numpy as np

class KNN:

    def __init__(self, k=1):
        self.k = k  # num of neighbors
        self.train_data_set = []    # [ Matrix1( num of sensors * timesteps ), ... ]
        self.train_label_set = []   # [ wafer label1, ... ]
        self.n_sensors = 83

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
    def get_uclidean_distance(self, mat1, mat2, per_column=False):

        diff = np.absolute(mat1-mat2)

        if per_column:
            dist = np.sum(diff, axis=0)
        else:
            dist = np.sum(diff)

        return dist

    # 각 Train data에 대해 Test data와의 거리를 구한다
    # 가장 가까운 k개의 data(neighbor)의 label을 반환한다
    # 가까움 측정에는 BORDA voting을 이용
    # TODO : implement weighted voting
    def get_neighbors_BORDA(self, train_data_set, train_label_set, test_data):
        distances = []
        length = len(train_data_set)
        for i in range(self.n_sensors):
            distances.append([])

        for i in range(length):
            dist = self.get_uclidean_distance(np.array(train_data_set[i]), np.array(test_data), per_column=True)

            for j in range(self.n_sensors): # self.n_sensors = len(dist)
                distances[j].append((dist[j], i))

        scores = [[0, i] for i in range(length)]
        for i in range(self.n_sensors):
            distances[i].sort()
            for j in range(len(distances[i])):
                scores[distances[i][j][1]][0] += j

        scores.sort()
        neighbors = []
        print('-----')
        for i in range(self.k):
            print(scores[i])
            # TODO : implement weighted voting
            neighbors.append(train_label_set[scores[i][1]])

        return np.array(neighbors)

    # 각 Train data에 대해 Test data와의 거리를 구한다
    # 가장 가까운 k개의 data(neighbor)의 label을 반환한다
    def get_neighbors(self, train_data_set, train_label_set, test_data):
        distances = []
        
        for i in range(len(train_data_set)):
            dist = self.get_uclidean_distance(np.array(train_data_set[i]), np.array(test_data))
            distances.append((dist, train_label_set[i]))

        distances.sort()
        neighbors = []
        print('-----')
        for i in range(self.k):
            print(distances[i][0], distances[i][1])
            neighbors.append(distances[i][1])

        return np.array(neighbors)

    # Test data(웨이퍼)의 양불을 추정하고 정확도를 확인한다.
    def test(self, test_data_set, test_label_set, method='UD'):
        predicted = self.predict(test_data_set, method=method)
        test_label_set = np.array(test_label_set)
        accuracy = 1 - (predicted != test_label_set).sum()/float(predicted.size)
        return test_label_set, predicted, accuracy

    # Test data(웨이퍼)의 양불을 추정한다.
    def predict(self, test_data_set, method='UD'):
        train_data_set, train_label_set = self.get_train_data()

        predicted = []
        for test_data in test_data_set:
            if method == 'UD':
                neighbors = self.get_neighbors(train_data_set, train_label_set, test_data)
            elif method == 'BORDA':
                neighbors = self.get_neighbors_BORDA(train_data_set, train_label_set, test_data)
            predicted.append(np.bincount(neighbors).argmax())

        return np.array(predicted)
