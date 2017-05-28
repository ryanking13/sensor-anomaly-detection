# -*- coding: utf-8 -*-

import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import fastdtw

class KNN:

    def __init__(self, k=1, distance_method='UD', neighbor_method='SIMPLE'):
        self.k = k  # num of neighbors
        self.train_data_set = []    # [ Matrix1( num of sensors * timesteps ), ... ]
        self.train_label_set = []   # [ wafer label1, ... ]
        self.n_sensors = 42

        self.distance_method = distance_method
        self.neighbor_method = neighbor_method

        # for Eros and PCA
        self.w_eigenvalues = np.zeros(1)

    #################################################
    # TRAIN METHODS (interface)
    #################################################

    # Load trained data to model
    # TODO: implement
    def load_train_data(self):
        return

    # return train data
    def get_train_data(self):
        return self.train_data_set, self.train_label_set

    # add new train data
    def add_train_data(self, train_data, train_label):

        method = self.distance_method

        if method == "UD":
            self.train_data_set.append(train_data)

        elif method == "Eros":
            self.train_data_set.append(self.eros_setup(train_data))

        elif method == "DTW":
            self.train_data_set.append(train_data)

        elif method == "FastDTW":
            self.train_data_set.append(train_data)

        elif method == "PCA_UD":
            self.train_data_set.append(self.pca_setup(train_data))

        elif method == "PCA_DTW":
            self.train_data_set.append(self.pca_setup(train_data))

        self.train_label_set.append(train_label)

    # Eros를 사용하기 위한 초기작업을 수행한다
    # 각 train data의 eigenvalue 값을 구한다
    # train data의 eigenvalue 값을 이용하여 곱해질 w를 구한다
    def eros_setup(self, data):

        V, s = self.get_svd(data)

        if self.w_eigenvalues.shape[0] != len(s):
            self.w_eigenvalus = np.zeros(len(s))

        for i in range(self.w_eigenvalues.shape[0]):
            self.w_eigenvalues[i] = max(self.w_eigenvalues[i], s[i])

        return V
    
    # 행렬의 svd 분석에서 나오는 eigenvalue 값을 구한다
    def get_svd(self, mat):
        U, s, V = np.linalg.svd(mat)

        sum_s = np.sum(s)
        s = np.divide(s, sum_s)

        return V, s

    def pca_setup(self, data):
        mat, eigenvalues = self.get_pca(data)
        
        if self.w_eigenvalues.shape[0] != len(eigenvalues):
            self.w_eigenvalues = np.zeros(len(eigenvalues))

        for i in range(self.w_eigenvalues.shape[0]):
            self.w_eigenvalues[i] = max(self.w_eigenvalues[i], eigenvalues[i])

        return mat

    def get_pca(self, data):
        pca = PCA(n_components=6)
        # data = StandardScaler().fit_transform(data)
        mat = pca.fit_transform(data)
        ev = pca.explained_variance_
        return mat, ev

    #################################################
    # TEST METHODS
    #################################################

    def test(self, test_data_set, test_label_set):

        predicted = self.predict(test_data_set)
        test_label_set = np.reshape(np.array(test_label_set), -1)
        accuracy = 1 - (predicted != test_label_set).sum() / float(predicted.size)
        return test_label_set, predicted, accuracy

    def predict(self, test_data_set):

        train_data_set, train_label_set = self.get_train_data()

        if self.distance_method == 'Eros':
            for i in range(len(test_data_set)):
                test_data_set[i], _ = self.get_svd(test_data_set[i])
        elif self.distance_method in ['PCA_UD', 'PCA_DTW']:
            for i in range(len(test_data_set)):
                test_data_set[i], _ = self.get_pca(test_data_set[i])

        predicted = []
        for test_data in test_data_set:
            if self.neighbor_method == 'SIMPLE':
                neighbors = self.get_neighbors(train_data_set, train_label_set, test_data)
            elif self.neighbor_method == 'BORDA':
                neighbors = self.get_neighbors_BORDA(train_data_set, train_label_set, test_data)

            predicted.append(np.bincount(neighbors).argmax())

        return np.array(predicted)


    #################################################
    # Distance methods
    #################################################

    # get Uclidean distance
    # assume size of two matrix is same ( if not, error )
    def get_uclidean_distance(self, mat1, mat2, per_column=False):

        diff = np.absolute(mat1-mat2)

        if per_column:
            dist = np.sum(diff, axis=0)
        else:
            dist = np.sum(diff)

        return dist

    # Eros(Extended Frobenius norm)을 이용한 distance를 계산한다
    # V1, V2 = right eigenvector matrix
    def get_eros_distance(self, V1, V2):
        w = self.w_eigenvalues

        mul = np.absolute(np.sum(np.multiply(V1, V2), axis=0))
        eros = 1 - np.sum(np.multiply(w, mul))

        return eros

    # get distance by Dynamic Time Warping method
    # assume size of two matrix is same ( if not, error )
    # very slow
    def get_dtw(self, mat1, mat2, per_column=False):

        global_cost_sum = 0
        cost_array = []
        num_sensors = mat1.shape[1]

        for col in range(num_sensors):
            # s1, s2 = sensor data vector
            s1 = mat1[:, col]
            s2 = mat2[:, col]

            # M != N 이어도 동작함
            M = len(s1)
            N = len(s2)
            cost = sys.maxsize * np.ones((M, N))

            # Initalize process
            cost[0, 0] = abs(s1[0] - s2[0])
            for i in range(1, M):
                cost[i, 0] = cost[i-1, 0] + abs(s1[i] - s2[0])
            for j in range(1, N):
                cost[0, j] = cost[0, j-1] + abs(s1[0] - s2[j])

            # Filling matrix
            for i in range(1, M):
                for j in range(1, N):
                    pre_cost = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                    cost[i, j] = min(pre_cost) + abs(s1[i] - s2[j])

            if per_column:
                cost_array.append(cost[-1,-1] / sum(cost.shape))
            else:
                global_cost_sum += cost[-1, -1] / sum(cost.shape)

        if per_column:
            return cost_array
        else:
            return global_cost_sum

    # get distance by Fase Dynamic Time Warping method
    # slow
    def get_fast_dtw(self, mat1, mat2):

        global_cost_sum = 0
        num_sensors = mat1.shape[1]

        for col in range(num_sensors):
            # s1, s2 = sensor data vector
            s1 = mat1[:, col]
            s2 = mat2[:, col]

            dist, _ = fastdtw.fastdtw(s1, s2, radius=1)
            global_cost_sum += dist

        return global_cost_sum

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
        n_dim = len(test_data[0])
        max_pointed = self.k + 10    # voting point를 지급할 개수
        weight = self.w_eigenvalues
        # weight = 1
        method = self.distance_method

        for i in range(n_dim):
            distances.append([])

        for i in range(length):
            if method == 'UD' or method == 'PCA_UD':
                dist = self.get_uclidean_distance(np.array(train_data_set[i]), np.array(test_data), per_column=True)
            elif method == 'Eros':
                print("IMPOSSIBLE COMBINATION")
                exit(-1)
            elif method == 'DTW' or method == 'PCA_DTW':
                dist = self.get_dtw(np.array(train_data_set[i]), np.array(test_data), per_column=True)

            for j in range(n_dim):
                distances[j].append((dist[j], i))

        scores = [[0, i] for i in range(length)]  # [score, train_data_num]
        for i in range(n_dim):
            distances[i].sort()
            d_p = distances[i][max_pointed][0] - distances[i][0][0]
            if d_p == 0:
                d_p = 1

            for j in range(max_pointed):
                d_j = distances[i][j][0] - distances[i][0][0]
                scores[distances[i][j][1]][0] += weight[i] * (1 + max_pointed*(1-(d_j/d_p)))

        scores.sort(reverse=True)
        neighbors = []
        # print('-----')
        for i in range(self.k):
            # print(scores[i])
            neighbors.append(train_label_set[scores[i][1]])

        return np.array(neighbors)

    # 각 Train data에 대해 Test data와의 거리를 구한다
    # 가장 가까운 k개의 data(neighbor)의 label을 반환한다
    def get_neighbors(self, train_data_set, train_label_set, test_data):

        distances = []
        method = self.distance_method

        # 각 Train data에 대하여 distance를 구한다
        for i in range(len(train_data_set)):
            dist = None

            if method == 'UD' or method == 'PCA_UD':
                dist = self.get_uclidean_distance(np.array(train_data_set[i]), np.array(test_data))
            elif method == 'Eros':
                dist = self.get_eros_distance(train_data_set[i], test_data)
            elif method == 'DTW' or method == 'PCA_DTW':
                dist = self.get_dtw(np.array(train_data_set[i]), np.array(test_data))
            elif method == 'FastDTW':
                dist = self.get_fast_dtw(np.array(train_data_set[i]), np.array(test_data))

            distances.append((dist, train_label_set[i], i)) # Third index is just for debug

        # 가장 작은 distance 순서대로 k개의 neighbor를 고른다
        distances.sort()
        neighbors = []
        # print('-----')
        for i in range(self.k):
            # print(distances[i][0], distances[i][1])
            neighbors.append(distances[i][1])

        return np.array(neighbors)
