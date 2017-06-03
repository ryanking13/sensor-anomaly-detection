from KNN import KNN
from Database import Database
import sys
from accuracy_measure import f1_score
import time

def find_min_len(mat1, mat2):
    min_len = 987654321
    for i in range(len(mat1)):
        min_len = min(len(mat1[i]), min_len)
    for i in range(len(mat2)):
        min_len = min(len(mat2[i]), min_len)

    return min_len

def cut(data_set, length):

    for i in range(len(data_set)):
        data_set[i] = data_set[i][:length]

    return data_set


def start_train(knn, dm, x, y, length=0):
    # print('[*] train start')

    if length > 0:
        x = cut(x, length)
    
    start_time = time.time()
    for i in range(len(x)):
        knn.add_train_data(x[i], y[i][0])
    end_time = time.time()

    return end_time - start_time
    # print('[*] train end')


def start_test(knn, dm, x, y, length=0):
    # print('[*] test start')

    if length > 0:
        x = cut(x, length)

    start_time = time.time()
    real, predicted, accuracy = knn.test(x, y)
    f1 = f1_score(real, predicted)
    end_time = time.time()

    # print('[*] test end')

    return real, predicted, accuracy, f1, end_time-start_time

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
    d_method = 'Eros'
    n_method = 'SIMPLE'
    knn = KNN(distance_method=d_method, neighbor_method=n_method)
    # print('[*] Done Construcing KNN model')

    train_data_set, train_label_set = dm.get_train_data(get_all=True)
    test_data_set, test_label_set = dm.get_test_data()

    min_len = find_min_len(train_data_set, test_data_set)
    train_time = start_train(knn, dm, train_data_set, train_label_set, length=min_len)
    
    real, predict, accuracy, f1, test_time = start_test(knn, dm, test_data_set, test_label_set, length=min_len)

    if True:
        print('   real: ', real)
        print('predict: ', predict)
        print('accuracy: %.6f' % accuracy)
        print('f1 score: %.6f' % f1)
        #print('%.6f' % f1)
        #print('time: %.6f %.6f' % (train_time, test_time))

    if False:
    
        f_accuracy = open("%s_%s_accuracy.txt" % (d_method, n_method), "a")
        f_f1 = open("%s_%s_fmeasure.txt" % (d_method, n_method), "a")
        f_train_time = open("%s_%s_traintime.txt" % (d_method, n_method), "a")
        f_test_time = open("%s_%s_testtime.txt" % (d_method, n_method), "a")
        
        postfix = ''
        try_size = '20'
        if try_num.endswith(try_size):
            postfix = '\n'
            print("done one set")

        f_accuracy.write("%.6f %s" % (accuracy, postfix))
        f_f1.write("%.6f %s" % (f1, postfix))
        f_train_time.write("%.6f %s" % (train_time, postfix))
        f_test_time.write("%.6f %s" % (test_time, postfix))

if __name__ == '__main__':
    main()
