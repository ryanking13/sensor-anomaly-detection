from KNN import KNN
from Database import Database
import sys

def start_train(knn, dm):
    print('[*] train start')
    train_data_set, train_label_set = dm.get_train_data(get_all=True)
    for i in range(len(train_data_set)):
        knn.add_train_data(train_data_set[i], train_label_set[i][0])
    print('[*] train end')


def start_test(knn, dm):
    x, y = dm.get_test_data();

    real, predicted, accuracy = knn.test(x, y, distance_method='UD', neighbor_method='SIMPLE')

    print('[*] test result')
    print('Real/Predicted(diff)');
    for i in range(len(real)):
        if real[i] != predicted[i]:
            print(real[i], predicted[i])
        elif real[i] == 0 and predicted[i] == 0:
            print('matched!')

    print("Accuracy : %.6f" % accuracy)

def main():
    
    try_num = sys.argv[1]
    step_num = '_step' + sys.argv[2]

    # try_num = '3'
    # step_num = '_step11'

    path = '../../data/'
    train_answer = 'trainList' + try_num + '.txt'
    test_answer = 'testList' + try_num + '.txt'
    print('[*] Loading data manager')
    dm = Database(train_path=path, train_answer_file=train_answer,
                test_path=path, test_answer_file=test_answer,
                sufix=step_num)
    print('[*] Done loading data manager')
    print('[*] Constructing KNN model')
    knn = KNN()
    print('[*] Done Construcing KNN model')

    start_train(knn, dm)
    start_test(knn, dm)

if __name__ == '__main__':
    main()
