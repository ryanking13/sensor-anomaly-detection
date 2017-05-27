from KNN import KNN
from Database import Database
import sys
from accuracy_measure import f1_score

def start_train(knn, dm):
    # print('[*] train start')
    train_data_set, train_label_set = dm.get_train_data(get_all=True)
    for i in range(len(train_data_set)):
        knn.add_train_data(train_data_set[i], train_label_set[i][0])
    # print('[*] train end')


def start_test(knn, dm):
    x, y = dm.get_test_data();

    # print('[*] test start')

    real, predicted, accuracy = knn.test(x, y)
    f1 = f1_score(real, predicted)

    # print('[*] test end')

    return real, predicted, accuracy, f1

def main():
    
    try_num = sys.argv[1]
    step_num = '_step' + sys.argv[2]

    # try_num = '3'
    # step_num = '_step11'

    path = '../../data/'
    train_answer = 'trainList' + try_num + '.txt'
    test_answer = 'testList' + try_num + '.txt'
    
    # print('[*] Loading data manager')
    dm = Database(train_path=path, train_answer_file=train_answer,
                test_path=path, test_answer_file=test_answer,
                sufix=step_num)
    # print('[*] Done loading data manager')
    
    # print('[*] Constructing KNN model')
    knn = KNN(distance_method='DTW', neighbor_method='BORDA')
    # print('[*] Done Construcing KNN model')

    start_train(knn, dm)
    
    real, predict, accuracy, f1 = start_test(knn, dm)

    #print('   real: ', real)
    #print('predict: ', predict)
    #print('accuracy: %.6f' % accuracy)
    print('f1 score: %.6f' % f1)

if __name__ == '__main__':
    main()
