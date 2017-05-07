from KNN import KNN
from LSTM_DataManager import DataManager


def start_train(knn, dm):
    print('[*] train start')
    train_data_set, train_label_set = dm.get_train_data(batch_size=0, get_all=True)
    for i in range(len(train_data_set)):
        # 데이터의 time length가 일치하지 않아서 임시 조치
        train_data_set[i] = train_data_set[i][:1750]
        knn.add_train_data(train_data_set[i], train_label_set[i])
    print('[*] train end')


def start_test(knn, dm):
    x, y = dm.get_test_data()

    # 데이터의 time length가 일치하지 않아서 임시 조치
    for i in range(len(x)):
        x[i] = x[i][:1750]

    real, predicted, accuracy = knn.test(x, y, distance_method='Eros', neighbor_method='SIMPLE')

    print('[*] test result')
    print("Pridicted :", predicted)
    print("Real :", real)
    print("Accuracy : %.6f" % accuracy)

def main():

    print('[*] Loading data manager')
    dm = DataManager()
    print('[*] Done loading data manager')
    print('[*] Constructing KNN model')
    knn = KNN()
    print('[*] Done Construcing KNN model')

    start_train(knn, dm)
    start_test(knn, dm)

if __name__ == '__main__':
    main()