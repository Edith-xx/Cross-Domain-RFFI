import h5py
import numpy as np
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform, randint
from sklearn.model_selection import train_test_split


def awgn(data):
    pkt_num = data.shape[0]
    #SNRdB = 20
    snr_range = np.arange(0, 11)
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)  # 产生pkt_num个信噪比值，范围在snr_range最小值和最大值之间
    #print(SNRdB)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        #SNR_linear = 10 ** (SNRdB/ 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        data[pktIdx] = s + n

    return data

class LoadDataset():
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)
        for i in range(0, num_col - 1, 2):
            data_complex[:, i // 2] = data[:, i] + 1j * data[:, i + 1]
        return data_complex

    def load_iq_samples(self, file_path, dev_range):

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)
        sample_index_list = []

        for dev_idx in dev_range:
            num_pkt = np.count_nonzero(label == dev_idx)
            pkt_range = np.arange(0, num_pkt, dtype=int)
            sample_index_dev = np.where(label == dev_idx)[0][
                pkt_range].tolist()  # 查找label中的标签值等于dev-idx，Where用于pkt_range里面提取对应的label的值
            sample_index_list.extend(sample_index_dev)
            print('Dev ' + str(dev_idx + 1) + ' have ' + str(num_pkt) + ' packets.')

        data = f[self.dataset_name][sample_index_list]  # 把data中的数据包和其标签对应起来
        data = self._convert_to_complex(data)
        label = label[sample_index_list]
        f.close()
        return data, label

def read_train_data(file_path='D:/work/data/paper3data/ft6/burst1.h5',
              dev_range = np.arange(0, 7, dtype=int)):

    data_all = []
    y_all = []

    LoadDatasetObj = LoadDataset()

    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range)
    data_ch0 = awgn(data_ch0)
    num_sample = len(data_ch0)
    real = np.real(data_ch0)
    imag = np.imag(data_ch0)
    merge_result = np.empty((num_sample, 2, 6000))
    merge_result[:, 0, :] = real
    merge_result[:, 1, :] = imag
    data_all.append(merge_result)
    y_all.append(y_ch0)
    data_all = np.concatenate(data_all, axis=0)
    y_all = np.concatenate(y_all)
    X_train, X_val, Y_train, Y_val = train_test_split(data_all, y_all, test_size=0.2, random_state=32)
    return X_train, X_val, Y_train, Y_val

#def read_test_data(file_path = 'F:/7identical drone/ft6/ft6_test.h5',
def read_test_data(file_path='D:/work/data/paper3data/ft6/burst4.h5',
              dev_range = np.arange(0, 7, dtype=int)):
    data_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()

    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                     dev_range)
    data_ch0 = awgn(data_ch0)
    num_sample = len(data_ch0)
    real = np.real(data_ch0)
    imag = np.imag(data_ch0)
    merge_result = np.empty((num_sample, 2, 6000))
    merge_result[:, 0, :] = real
    merge_result[:, 1, :] = imag
    data_all.append(merge_result)
    y_all.append(y_ch0)
    X_test = np.concatenate(data_all, axis=0)
    print(X_test.shape)
    Y_test = np.concatenate(y_all)
    return X_test, Y_test
if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = read_train_data()
    X_test, Y_test = read_test_data()








