#是原始的没加2DDWT的结果
import h5py
import numpy as np
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform, randint
import torchvision
from scipy import signal
from torch import nn
from sklearn.model_selection import train_test_split

def awgn(data):
    pkt_num = data.shape[0]
    SNRdB = 1
    #snr_range = np.arange(0, 11)
    #SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)  # 产生pkt_num个信噪比值，范围在snr_range最小值和最大值之间
    #print(SNRdB)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        #SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        SNR_linear = 10 ** (SNRdB/ 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        data[pktIdx] = s + n

    return data
class STFTSpectrogram():
    def __init__(self, ):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm


    def _spec_crop(self, x):
        num_row = x.shape[0]
        x_cropped = x[round(num_row * 0.3):round(num_row * 0.7)]

        return x_cropped

    def gen_stft_spectrogram(self, sig, win_len= 256, overlap= 128):
        f, t, spec = signal.stft(sig,
                                 window='boxcar',
                                 nperseg=win_len,  #窗函数长度
                                 noverlap=overlap,
                                 nfft=win_len,    #FFT长度，默认是窗函数长度，如果大于窗函数长度，会自动进行零填充
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)
        spec = np.fft.fftshift(spec, axes=0)
        spec = np.log10(np.abs(spec) ** 2)
        return spec

    def STFT_spectrogram(self, data):
        data = self._normalization(data)
        #data.shape=[样本个数，95463每个样本里面的IQ信号个数]
        num_sample = data.shape[0]
        #num_row = 256
        num_row = int(256 * 0.4)

        num_column = int(np.floor((data.shape[1] - 256) / 128)+1)

        data_dspec = np.zeros([num_sample, 1, num_row, num_column])  #1表示谱图是单通道的，torch中channel在前面
        #在keras中是(num_samples, num_freq_bins, num_time_bins, channel)
        for i in range(num_sample):
            stft_spectrogram = self.gen_stft_spectrogram(data[i])
            stft_spectrogram = self._spec_crop(stft_spectrogram)

            data_dspec[i, 0, :, :] = stft_spectrogram          #直接返回傅里叶谱图

        return data_dspec

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
#def read_train_data(file_path = 'F:/7identical drone/ft6/ft6_train.h5',
def read_train_data():
    data_stft_all = []
    y_all = []
    snr_range = np.arange(0, 11)
    file_path='D:/work/data/paper3_data/paper3data1/ft6/burst123.h5'
    dev_range = np.arange(0, 7, dtype=int)
    LoadDatasetObj = LoadDataset()
    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path, dev_range)
    STFTSpectrogramObj = STFTSpectrogram()
    data_stft = STFTSpectrogramObj.STFT_spectrogram(data_ch0)

    data_stft_all.append(data_stft)
    y_all.append(y_ch0)
    data_stft_all = np.concatenate(data_stft_all, axis = 0)
    new_shape = (data_stft_all.shape[0], data_stft_all.shape[2], data_stft_all.shape[1] * data_stft_all.shape[3])
    data_stft_all = data_stft_all.reshape(new_shape)
    data_stft_all = data_stft_all.transpose(0, 2, 1)
    y_all = np.concatenate(y_all, axis = 0)
    X_train, X_val, Y_train, Y_val = train_test_split(data_stft_all, y_all, test_size=0.2, random_state=32)
    return X_train, X_val, Y_train, Y_val

def read_test_data():
    file_path = 'D:/work/data/paper3_data/paper3data1/ft6/burst4.h5'
    dev_range = np.arange(0, 7, dtype=int)
    data_stft_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()
    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path, dev_range)
    STFTSpectrogramObj = STFTSpectrogram()
    data_stft = STFTSpectrogramObj.STFT_spectrogram(data_ch0)
    data_stft_all.append(data_stft)
    y_all.append(y_ch0)
    X_test = np.concatenate(data_stft_all, axis=0)
    new_shape = (X_test.shape[0], X_test.shape[2], X_test.shape[1] * X_test.shape[3])
    X_test = X_test.reshape(new_shape)
    X_test = X_test.transpose(0, 2, 1)
    Y_test = np.concatenate(y_all)
    return X_test, Y_test
if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = read_train_data()
    X_test, Y_test = read_test_data()







