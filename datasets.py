import mne
import sys
from Feature_Extraction import Feature_Extraction
from Preprocessing import Preprocessing
import numpy as np
from read_data_tuarv2 import ReadDataTUARv2, EDFDataTUARv2
from read_data_motor_imaginary import ReadDataMotorImaginary

def select_ch_df(df):
    return df[df[1].str.contains("FP")]


def mean_std_TUARv2(raw_time):
    features = []

    for raw in raw_time:
        data = raw._data
        mean0 = np.mean(data[0])
        sd0 = np.std(data[0])
        mean1 = np.mean(data[1])
        sd1 = np.std(data[1])
        features.append([mean0, sd0, mean1, sd1])

    return features


def TUARv2():
    # use gpu cuda cores
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    mne.cuda.init_cuda(verbose=True)
    # channels to work on: FP1, FP2

    edf_01_tcp_ar = ReadDataTUARv2("TUAR/v2.0.0/lists/edf_01_tcp_ar.list",
                              "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                              "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

    preprocessing = Preprocessing()

    raw_ch = []
    ranges = []
    edf_01_tcp_ar.labels = select_ch_df(edf_01_tcp_ar.labels)
    # 'EEG FP1-REF', 'EEG FP2-REF'
    for data in edf_01_tcp_ar.data:
        raw, name = data
        raw = preprocessing.select_ch(raw)
        ranges.append(edf_01_tcp_ar.labels[edf_01_tcp_ar.labels[0] == name])
        raw_ch.append(raw)

    del edf_01_tcp_ar

    # 'EEG FP1-LE', 'EEG FP2-LE'

    edf_02_tcp_le = ReadDataTUARv2("TUAR/v2.0.0/lists/edf_02_tcp_le.list",
                              "TUAR/v2.0.0/csv/labels_02_tcp_le.csv",
                              "TUAR/v2.0.0/_DOCS/02_tcp_le_montage.txt").get_data()
    edf_02_tcp_le.labels = select_ch_df(edf_02_tcp_le.labels)
    for data in edf_02_tcp_le.data:
        raw, name = data
        raw = preprocessing.select_ch(raw, ch_names=['EEG FP1-LE', 'EEG FP2-LE'])
        label = edf_02_tcp_le.labels[edf_02_tcp_le.labels[0] == name]
        raw_ch.append(raw)
        ranges.append(label)

    del edf_02_tcp_le

    edf_03_tcp_ar_a = ReadDataTUARv2("TUAR/v2.0.0/lists/edf_03_tcp_ar_a.list",
                                "TUAR/v2.0.0/csv/labels_03_tcp_ar_a.csv",
                                "TUAR/v2.0.0/_DOCS/03_tcp_ar_a_montage.txt").get_data()

    edf_03_tcp_ar_a.labels = select_ch_df(edf_03_tcp_ar_a.labels)
    # 'EEG FP1-REF', 'EEG FP2-REF'
    for data in edf_03_tcp_ar_a.data:
        raw, name = data
        raw = preprocessing.select_ch(raw)
        label = edf_03_tcp_ar_a.labels[edf_03_tcp_ar_a.labels[0] == name]
        raw_ch.append(raw)
        ranges.append(label)

    del edf_03_tcp_ar_a

    raw_time = []
    labels = []

    for i in range(0, len(raw_ch)):
        print(i + 1, '/', len(raw_ch))
        if i == 18 or i == 29 or i == 67 or i == 68 or i == 129 or i == 131 or i == 136 or i == 175 or i == 178:
            continue
        elif i == 179 or i == 181 or i == 224:
            continue
        raw = raw_ch[i]
        df = ranges[i]
        for j in df.index:
            raw_time.append(preprocessing.get_time_range_raw(raw, start_time=df.loc[j, 2], end_time=df.loc[j, 3]))
            labels.append(EDFDataTUARv2.LABELS_MAP_NAME_NUMBER[df.loc[j, 4]])

    del raw_ch
    del ranges

    features = mean_std_TUARv2(raw_time)

    del raw_time

    m = {14: 0, 21: 1, 22: 2, 23: 3, 30: 4, 100: 1, 101: 1, 102: 1, 103: 1, 105: 1, 106: 1, 109: 1}

    for i in range(0, len(labels)):
        labels[i] = int(labels[i])
        labels[i] = m[labels[i]]

    n_output = set(labels)
    n_output = len(n_output)

    print(sys.getsizeof(features), sys.getsizeof(labels))

    np.save('features.tuar/features_mean_std.npy', features)
    np.save('features.tuar/labels.npy', labels)

    return features, labels, n_output


def load_processed_features_TUARv2(path_features='features.tuar/features_mean_std.npy'):
    features = np.load(path_features)
    labels = np.load('features.tuar/labels.npy')
    n_output = set(labels)
    n_output = len(n_output)

    return features, labels, n_output


def motor_imaginary():
    data, names = ReadDataMotorImaginary().get_data()
    markers_f5 = []
    signals_f5 = []
    # electrodes by col.
    # Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 A1 A2 F7 F8 T3 T4 T5 T6 Fz Cz Pz X3
    # 5F data
    for i in range(0, 19):
        print('Get important arrays from 5F data: ', i + 1, '/', 19)
        d = data[i]
        marker = []
        for m in d['o'][0][0][4]:
            marker.append(int(m[0]))
        markers_f5.append(marker)
        signals_f5.append(d['o'][0][0][5])

    data_f5 = []
    for i in range(len(signals_f5)):
        print('Separate channels for 5F data: ', i + 1, '/', len(signals_f5))
        signal = signals_f5[i]
        FP1 = []
        FP2 = []
        marker = []
        for j in range(len(signal)):
            reading = signal[j]
            fp1 = reading[0]
            fp2 = reading[1]
            FP1.append(fp1)
            FP2.append(fp2)
            marker.append(markers_f5[i][j])
        data_f5.append([FP1, FP2, marker])
    del signals_f5, markers_f5

    # general data
    markers = []
    signals = []
    for i in range(22, len(data)):
        print('Get important arrays from general data: ', i + 1, '/', len(data))
        d = data[i]
        marker = []
        for m in d['o'][0][0][4]:
            marker.append(int(m[0]))
        markers.append(marker)
        signals.append(d['o'][0][0][5])
    del data

    data = []
    for i in range(len(signals)):
        print('Separate channels for general data: ', i + 1, '/', len(signals))
        signal = signals[i]
        FP1 = []
        FP2 = []
        marker = []
        for j in range(len(signal)):
            reading = signal[j]
            fp1 = reading[0]
            fp2 = reading[1]
            FP1.append(fp1)
            FP2.append(fp2)
            marker.append(markers[i][j])
        data.append([FP1, FP2, marker])
    del signals, markers

    data_f5 = np.array(data_f5, dtype=object)
    data = np.array(data, dtype=object)
    np.save('features.motor_dataset/raw_data_f5.npy', data_f5)
    print('saved F5 data')
    np.save('features.motor_dataset/raw_data_general.npy', data)
    print('saved general data')

    return data_f5, data


def load_raw_motor_dataset_data():
    data_f5 = np.load('features.motor_dataset/raw_data_f5.npy', allow_pickle=True)
    print('loaded F5 data')
    data = np.load('features.motor_dataset/raw_data_general.npy', allow_pickle=True)
    print('loaded general data')

    return data_f5, data


def segment_motor_data(data, labels, mapping, path):
    print('Making: ', path)
    seg = []
    progress = 0
    for file in data:
        progress += 1
        print(progress, '/', len(data))
        cur = 0
        temp_fp1 = []
        temp_fp2 = []
        fp1, fp2, marker = file[0], file[1], file[2]
        for i in range(len(fp1)):
            if cur == 0 and marker[i] not in labels:
                continue
            elif cur == 0 and marker[i] in labels:
                temp_fp1.append(fp1[i])
                temp_fp2.append(fp2[i])
                cur = marker[i]
            elif cur != 0 and marker[i] not in labels:
                seg.append((temp_fp1, temp_fp2, mapping[cur]))
                temp_fp1 = []
                temp_fp2 = []
                cur = 0
            elif cur != 0 and marker[i] == cur:
                temp_fp1.append(fp1[i])
                temp_fp2.append(fp2[i])
            elif cur != 0 and marker[i] != cur and marker[i] in labels:
                seg.append((temp_fp1, temp_fp2, mapping[cur]))
                temp_fp1 = []
                temp_fp2 = []
                temp_fp1.append(fp1[i])
                temp_fp2.append(fp2[i])
                cur = marker[i]

        if cur != 0:
            seg.append((temp_fp1, temp_fp2, mapping[cur]))

    seg = np.array(seg, dtype=object)
    np.save(path, seg)
    return seg


def load_seg_motor_dataset():
    seg_f5 = np.load('features.motor_dataset/seg_data_f5.npy', allow_pickle=True)
    print('loaded segmented F5 data')
    seg_gen = np.load('features.motor_dataset/seg_data_gen.npy', allow_pickle=True)
    print('loaded segmented general data')
    return seg_f5, seg_gen


def get_features_motor_dataset(data, set_labels, path_features, path_labels, path_set_labels):
    features = []
    labels = []
    n_output = len(set_labels)
    for reading in data:
        fp1, fp2, label = reading[0], reading[1], reading[2]
        mean1 = np.mean(fp1)
        std1 = np.std(fp1)
        mean2 = np.mean(fp2)
        std2 = np.std(fp2)
        features.append([mean1, std1, mean2, std2])
        labels.append(label)
    np.save(path_features, features)
    np.save(path_labels, labels)
    np.save(path_set_labels, set_labels)
    return features, labels, n_output


def load_features_motor_dataset(path_features, path_labels, path_set_labels):
    features = np.load(path_features)
    labels = np.load(path_labels)
    n_output = np.load(path_set_labels)
    n_output = len(n_output)
    return features, labels, n_output

