# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

class KNN:

    def __init__(self, k=1):
        self.k = k  # num of neighbors
        self.train_data_set = []    # [ Matrix1( num of sensors * timesteps ), ... ]
        self.train_label_set = []   # [ wafer label1, ... ]
        self.n_sensors = 83

        # for Eros
        self.train_eigenvectors = np.array([])
        self.w_eigenvalues = np.array([])

    #################################################
    # TRAIN METHODS (interface)
    #################################################

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


    #################################################
    # TEST METHODS
    #################################################

    # Test data(웨이퍼)의 양불을 추정하고 정확도를 확인한다.
    def test(self, test_data_set, test_label_set, distance_method='UD', neighbor_method = 'SIMPLE'):

        predicted = self.predict(test_data_set, distance_method=distance_method, neighbor_method=neighbor_method)
        test_label_set = np.array(test_label_set)
        accuracy = 1 - (predicted != test_label_set).sum() / float(predicted.size)
        return test_label_set, predicted, accuracy

    # Test data(웨이퍼)의 양불을 추정한다.
    def predict(self, test_data_set, distance_method='UD', neighbor_method = 'SIMPLE'):

        if distance_method == 'Eros':
            self.eros_setup()

        train_data_set, train_label_set = self.get_train_data()

        predicted = []
        for test_data in test_data_set:
            if neighbor_method == 'SIMPLE':
                neighbors = self.get_neighbors(train_data_set, train_label_set, test_data, method=distance_method)
            elif neighbor_method == 'BORDA':
                neighbors = self.get_neighbors_BORDA(train_data_set, train_label_set, test_data)
            predicted.append(np.bincount(neighbors).argmax())

        return np.array(predicted)


    #################################################
    # Distance methods
    #################################################

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

    # 행렬의 svd 분석에서 나오는 eigenvalue 값을 구한다
    def get_eigenvalues(self, mat):
        U, s, V = np.linalg.svd(mat)

        sum_s = np.sum(s)
        s = np.divide(s, sum_s)

        return V, s

    # Eros(Extended Frobenius norm)을 이용한 similarity를 계산한다
    def get_eros_distance(self, V1, V2):
        w = self.w_eigenvalues

        mul = np.absolute(np.sum(np.multiply(V1, V2), axis=0))
        eros = 1 - np.sum(np.multiply(w, mul))

        return eros

    # Eros를 사용하기 위한 초기작업을 수행한다
    # 각 train data의 eigenvalue 값을 구한다
    # train data의 eigenvalue 값을 이용하여 곱해질 w를 구한다
    def eros_setup(self, w_method='average'):
        print("[*] Staring Eros setup...")
        data, _ = self.get_train_data()

        # 각 Train data에 대한 eigenvalue를 구한다
        V_rows = []
        s_rows = []
        for d in data:
            V, s = self.get_eigenvalues(d)
            V_rows.append(V)
            s_rows.append(s)

        self.train_eigenvectors = np.array(V_rows)

        # 곱해질 w를 구한다
        # TODO : reimplement w_
        if w_method == 'max':
            self.w_eigenvalues = np.max(np.array(s_rows), axis=0)
        elif w_method == 'min':
            self.w_eigenvalues = np.min(np.array(s_rows), axis=0)
        elif w_method == 'average':
            self.w_eigenvalues = np.average(np.array(s_rows), axis=0)

        print("[*] Ended Eros setup")
        # self.w_eigenvalues = np.ones(self.w_eigenvalues.shape) # TEMP
        return self.w_eigenvalues

    #################################################
    # Neighbor Methods
    #################################################

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
    def get_neighbors(self, train_data_set, train_label_set, test_data, method='UD'):
        distances = []

        V2 = None
        if method == 'Eros':
            # Test data의 eigenvalue를 구한다
            V2, _ = self.get_eigenvalues(test_data)

        # 각 Train data에 대하여 distance를 구한다
        for i in range(len(train_data_set)):
            dist = None

            if method == 'UD':
                dist = self.get_uclidean_distance(np.array(train_data_set[i]), np.array(test_data))
            elif method == 'Eros':
                V1 = self.train_eigenvectors[i]
                dist = self.get_eros_distance(V1, V2)

            distances.append((dist, train_label_set[i], i)) # Third index is just for debug

        # 가장 작은 distance 순서대로 k개의 neighbor를 고른다
        distances.sort()
        neighbors = []
        print('-----')
        for i in range(self.k):
            print(distances[i][0], distances[i][1])
            neighbors.append(distances[i][1])

        return np.array(neighbors)
