import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from DataManager import DataManager
from LSTMnetwork import LSTMNetwork


def start_train(batch_size, dm, network, epoch_size=20):

    print('[*] train start')
    for epoch in range(epoch_size):
        x_batch, y_batch = dm.get_train_data(batch_size)

        # 데이터의 time length가 일치하지 않아서 임시 조치
        for i in range(len(x_batch)):
            x_batch[i] = x_batch[i][:1750]

        print(str(epoch) + ': ', end='')
        network.train(x_batch, y_batch)

    print('[*] train end')


def start_test(dm, network):
    x, y = dm.get_test_data()

    # 데이터의 time length가 일치하지 않아서 임시 조치
    for i in range(len(x)):
        x[i] = x[i][:1750]

    real, predict, accuracy = network.predict(x, y)

    print('[*] test result')
    print('Real Value: ', real)
    print('Predicted Value: ', predict)
    print('Accuracy: ', accuracy)


def main():
    batch_size_dft = 10
    test_size_dft = None

    batch_size = input("Batch Size (Default = 10) : ")
    test_size = input("Test Size (Default = Total Train data / 5 ) :  ")

    if batch_size == '':
        batch_size = batch_size_dft
    if test_size == '':
        test_size = test_size_dft

    print('[*] Loading data manager')
    dm = DataManager(num_test_data=test_size_dft)
    print('[*] Done loading data manager')

    print('[*] Constructing network')
    network = LSTMNetwork()
    print('[*] Done constructing network')

    start_train(batch_size_dft, dm, network)
    start_test(dm, network)

if __name__ == '__main__':
    main()