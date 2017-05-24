from KNN import KNN
from Database import Database


def start_train(knn, dm):
    print('[*] train start')
    train_data_set, train_label_set = dm.get_train_data(batch_size=0, get_all=True)
    for i in range(len(train_data_set)):
        knn.add_train_data(train_data_set[i], train_label_set[i])
    print('[*] train end')


def start_test(knn, dm):
    x, y = dm.get_test_data();

    real, predicted, accuracy = knn.test(x, y, distance_method='Eros', neighbor_method='SIMPLE')

    print('[*] test result')
    print('Real/Predicted(diff)');
    for i in range(len(real)):
        if real[i] != predicted[i]:
            print(real[i], predicted[i])
        elif real[i] == 0 and predicted[i] == 0:
            print('matched!')

    print("Accuracy : %.6f" % accuracy)

def main():

    print('[*] Loading data manager')
    dm = Database(num_test_data=200)
    print('[*] Done loading data manager')
    print('[*] Constructing KNN model')
    knn = KNN()
    print('[*] Done Construcing KNN model')

    start_train(knn, dm)
    start_test(knn, dm)

if __name__ == '__main__':
    main()
