import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from Database import Database
from LSTMnetwork import LSTMNetwork
from accuracy_measure import f1_score
import random, time

def cut(data_set):
    min_len = 987654321
    for i in range(len(data_set)):
        min_len = min( len(data_set[i]), min_len)

    for i in range(len(data_set)):
        data_set[i] = data_set[i][:min_len]

    return data_set

def start_train(batch_size, dm, network, epoch_size=100):

    # print('[*] train start')
    start_time = time.time()

    for epoch in range(epoch_size):
        x_batch, y_batch = dm.get_train_data(batch_size)
        x_batch = cut(x_batch)
        #print(str(epoch) + ': ', end='')
        network.train(x_batch, y_batch)

    end_time = time.time()
    # print('[*] train end')

    return end_time - start_time


def start_test(dm, network):
    x, y = dm.get_test_data()
    x = cut(x)
    # print('[*] test start')
    start_time = time.time()

    real, predict, accuracy = network.test(x, y)
    f1 = f1_score(real, predict)

    end_time = time.time()
    # print('[*] test end')

    return real, predict, accuracy, f1, end_time-start_time

def main():
    # batch_size_dft = 10
    #
    # batch_size = input("Batch Size (Default = 10) : ")
    #
    # if batch_size == '':
    #     batch_size = batch_size_dft

    # try_num = sys.argv[1]
    # step_num = '_step' + sys.argv[2]

    # print(try_num, step_num)
    try_num = str(random.randint(1, 20))
    step_num = '_step11'

    batch_size = 65
    path = '../../data/'
    train_answer = 'trainList' + try_num + '.txt'
    test_answer = 'testList' + try_num + '.txt'

    # print('[*] Loading data manager')
    dm = Database(train_path=path, train_answer_file=train_answer,
                  test_path=path, test_answer_file=test_answer,
                  sufix=step_num)
    # print('[*] Done loading data manager')

    # print('[*] Constructing network')
    network = LSTMNetwork()
    # print('[*] Done constructing network')

    train_time = start_train(batch_size, dm, network)
    real, predict, accuracy, f1, test_time = start_test(dm, network)

    # print("----RESULTS----")
    # print('train_time: ', train_time)
    # print('test_time: ', test_time)

    print('   real: ', real)
    print('predict: ', predict)
    print('accuracy: %.6f' % accuracy)
    print('f1 score: %.6f' % f1)


if __name__ == '__main__':
    main()
