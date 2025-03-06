#with 2D-DWT
import h5py
import torch
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
import pywt
from PIL import Image

class STFTSpectrogram():
    def __init__(self, ):
        pass

    def wavelet_transform(self, image):
        coeffs = pywt.dwt2(image[:, :], 'haar')
        cA, (cH, cV, cD) = coeffs
        #wavelet_result = np.stack((cA, cV, cH, cD), axis=0)
        return cA, cV, cH, cD

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
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]

        chan_ind_spec_amp = np.log10(np.abs(spec) ** 2)

        return chan_ind_spec_amp

    def STFT_spectrogram(self, data):
        data = self._normalization(data)
        num_sample = data.shape[0]
        data_dspec = np.zeros([num_sample, 1, 102, 45])
        A = V = H = D = np.zeros([num_sample, 1, 51, 23])
        for i in range(num_sample):
            stft_spectrogram = self.gen_stft_spectrogram(data[i])
            stft_spectrogram = self._spec_crop(stft_spectrogram)
            data_dspec[i, 0, :, :] = stft_spectrogram          #直接返回傅里叶谱图

            A[i, 0, :, :], V[i, 0, :, :], H[i, 0, :, :], D[i, 0, :, :] = self.wavelet_transform(data_dspec[i, 0, :, :])
        merged_result = np.empty((num_sample, 4, 51, 23))

        # 将四个变量按照第二个维度合并
        merged_result[:, 0, :, :] = A[:, 0, :, :]
        merged_result[:, 1, :, :] = V[:, 0, :, :]
        merged_result[:, 2, :, :] = H[:, 0, :, :]
        merged_result[:, 3, :, :] = D[:, 0, :, :]
        return merged_result

class LoadDataset():
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):

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
def read_train_data(file_path='D:/work/data/paper3_data/paper3data1/ft9/100/burst123.h5',
              dev_range = np.arange(0, 7, dtype=int)):

    data_stft_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()

    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range)

    STFTSpectrogramObj = STFTSpectrogram()
    data_stft = STFTSpectrogramObj.STFT_spectrogram(data_ch0)

    data_stft_all.append(data_stft)
    y_all.append(y_ch0)

    data_stft_all = np.concatenate(data_stft_all, axis=0)
    new_shape = (data_stft_all.shape[0], data_stft_all.shape[2], data_stft_all.shape[1] * data_stft_all.shape[3])
    data_stft_all = data_stft_all.reshape(new_shape)
    data_stft_all = data_stft_all.transpose(0, 2, 1)
    print(data_stft_all.shape)
    y_all = np.concatenate(y_all)
    X_train, X_val, Y_train, Y_val = train_test_split(data_stft_all, y_all, test_size=0.2, random_state=32)
    return X_train, X_val, Y_train, Y_val

def read_test_data(file_path='D:/work/data/paper3_data/paper3data1/ft9/100/burst4.h5',
              dev_range = np.arange(0, 7, dtype=int)):
    data_stft_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()

    data_ch0, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                     dev_range)

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





