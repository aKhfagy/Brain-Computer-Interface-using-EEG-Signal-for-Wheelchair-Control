import mne
import sys
import numpy as np
import antropy as ant
import pywt
from read_data_tuarv2 import ReadDataTUARv2, EDFDataTUARv2
from read_data_motor_imaginary import ReadDataMotorImaginary

PATH_TUARv2_C3_C4 = 'features.tuar/features_mean_std_c3_c4.npy'
PATH_TUARv2_FP1_FP2 = 'features.tuar/features_mean_std_fp1_fp2.npy'
def select_ch_df(df):
    return df[df[1].str.contains("FP")]


def select_ch(raw, ch_names):
    raw = raw.pick_channels(ch_names, ordered=False)
    return raw


def get_time_range_raw(raw, start_time, end_time):
    raw_select = raw.copy().crop(tmin=start_time, tmax=end_time)
    return raw_select


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
    path = PATH_TUARv2_FP1_FP2
    # use gpu cuda cores
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    mne.cuda.init_cuda(verbose=True)
    # channels to work on: FP1, FP2

    edf_01_tcp_ar = ReadDataTUARv2("TUAR/v2.0.0/lists/edf_01_tcp_ar.list",
                              "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                              "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

    raw_ch = []
    ranges = []
    edf_01_tcp_ar.labels = select_ch_df(edf_01_tcp_ar.labels)
    # 'EEG FP1-REF', 'EEG FP2-REF'
    for data in edf_01_tcp_ar.data:
        raw, name = data
        raw = select_ch(raw, ['EEG FP1-REF', 'EEG FP2-REF'])
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
        raw = select_ch(raw, ['EEG FP1-LE', 'EEG FP2-LE'])
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
        raw = select_ch(raw, ['EEG FP1-REF', 'EEG FP2-REF'])
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
            raw_time.append(get_time_range_raw(raw, start_time=df.loc[j, 2], end_time=df.loc[j, 3]))
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

    np.save(path, features)
    np.save('features.tuar/labels.npy', labels)

    return features, labels, n_output


def segment_motor_data(data, labels, mapping, subject):
    print('Segmenting for subject:', subject)
    seg = []
    progress = 0
    for file in data:
        progress += 1
        print('File:', progress, '/', len(data))
        cur = 0
        temp = []
        ch_len = len(file[0])
        marker = file[1]
        for i in range(ch_len):
            if cur == 0 and marker[i] not in labels:
                continue
            elif cur == 0 and marker[i] in labels:
                temp.append(file[0][i])
                cur = marker[i] if isinstance(marker[i], int) else marker[i][0]
            elif cur != 0 and marker[i] not in labels:
                for seg_i in range(0, len(temp) - 200, 200):
                    x = temp[seg_i: seg_i + 200]
                    seg.append([x, mapping[cur]])
                temp = []
                cur = 0
            elif cur != 0 and marker[i] == cur:
                temp.append(file[0][i])
            elif cur != 0 and marker[i] != cur and marker[i] in labels:
                for seg_i in range(0, len(temp) - 200, 200):
                    x = temp[seg_i: seg_i + 200]
                    seg.append([x, mapping[cur]])
                temp = []
                temp.append(file[0][i])
                cur = marker[i] if isinstance(marker[i], int) else marker[i][0]

        if cur != 0:
            for seg_i in range(0, len(temp) - 200, 200):
                x = temp[seg_i: seg_i + 200]
                seg.append([x, mapping[cur]])

    get_features_motor_dataset(seg, set_labels=labels,
                               path_features='features.motor_dataset/data_features' + subject + '.npy',
                               path_labels='features.motor_dataset/data_labels' + subject + '.npy',
                               path_set_labels='features.motor_dataset/data_set_labels' + subject + '.npy')

    wavelet_processing(seg, path_data='features.motor_dataset/data_wavelet' + subject + '.npy')


def motor_imaginary():
    data, names = ReadDataMotorImaginary().get_data()
    markers = []
    signals = []
    # electrodes by col.
    # Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 A1 A2 F7 F8 T3 T4 T5 T6 Fz Cz Pz X5
    #  0   1   2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
    c4 = 4
    for i in range(0, len(data)):
        print('Get data and markers:', i + 1, '/', 19)
        d = data[i]
        markers.append(d['o'][0][0][4])
        signals.append(d['o'][0][0][5])
    del data
    subjects = ['A', 'B', 'C', 'D', 'E', 'F']
    subj = {}
    for i in range(len(subjects)):
        subj[subjects[i]] = i
    data_f5 = {}
    for subject in subjects:
        data_f5[subj[subject]] = []
    for i in range(len(signals)):
        print('Separating channels:', i + 1, '/', len(signals))
        signal = signals[i]
        ch = []
        marker = []
        for j in range(len(signal)):
            reading = signal[j]
            ch.append(reading[c4])
            marker.append(markers[i][j])
        data_f5[subj[names[i][10]]].append([ch, marker])
    del signals, markers
    index = [1, 2, 3, 4, 5, 6, 91, 92, 99]
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 91: 6, 92: 7, 99: 8}
    print('Segmenting data each second (200 readings)')
    for subject in subjects:
        segment_motor_data(data_f5[subj[subject]], index, mapping, subject)
    del data_f5


def get_features_motor_dataset(data, set_labels, path_features, path_labels, path_set_labels):
    print('Processing for:', path_features)
    features = []
    labels = []
    for reading in data:
        label = reading[1]
        wave = reading[0]
        mean = np.mean(wave)
        std = np.std(wave)
        median = np.median(wave)
        variance = np.var(wave)
        entropy = ant.svd_entropy(wave, normalize=True)
        hjorth = ant.hjorth_params(wave)
        features.append([mean, median, variance, std, entropy, hjorth[0], hjorth[1]])
        labels.append(label)
    print('Making:', path_features)
    np.save(path_features, features)
    print('Making:', path_labels)
    np.save(path_labels, labels)
    print('Making:', path_set_labels)
    np.save(path_set_labels, set_labels)


def wavelet_processing(data, path_data):
    print('Making:', path_data)
    w = pywt.Wavelet('db1')
    levels = pywt.dwt_max_level(data_len=200, filter_len=w.dec_len)
    wavelet = []
    for reading in data:
        wave = reading[0]
        coefficient = pywt.wavedec(wave, 'db1', level=levels)
        wavelet.append(coefficient)

