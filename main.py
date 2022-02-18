# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:07:13 2021

@author: Ahmed
"""
import os
import sys

import mne
from Read_Data import Read_Data, EDF_Data
from Feature_Extraction import Feature_Extraction
from Preprocessing import Preprocessing
import numpy as np
from FNN import FNN

# use gpu cuda cores
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)


def select_ch_df(df):
    return df[df[1].str.contains("FP")]


# channels to work on: FP1, FP2

edf_01_tcp_ar = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
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

edf_02_tcp_le = Read_Data("TUAR/v2.0.0/lists/edf_02_tcp_le.list",
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

edf_03_tcp_ar_a = Read_Data("TUAR/v2.0.0/lists/edf_03_tcp_ar_a.list",
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
    print (i, '/', len(raw_ch))
    if i == 18 or i == 29 or i == 67 or i == 68 or i == 129 or i == 131 or i == 136 or i == 175 or i == 178:
        continue
    elif i == 179 or i == 181 or i == 224:
        continue
    raw = raw_ch[i]
    df = ranges[i]
    for j in df.index:
        raw_time.append(preprocessing.get_time_range_raw(raw, start_time=df.loc[j, 2], end_time=df.loc[j, 3]))
        labels.append(EDF_Data.LABELS_MAP_NAME_NUMBER[df.loc[j, 4]])

del raw_ch
del ranges

features = []

for raw in raw_time:
    data = raw._data
    mean0 = np.mean(data[0])
    sd0 = np.std(data[0])
    mean1 = np.mean(data[1])
    sd1 = np.std(data[1])
    features.append([mean0, sd0, mean1, sd1])

del raw_time

m = {14: 0, 21: 1, 22: 2, 23: 3, 30: 4, 100: 1, 101: 1, 102: 1, 103: 1, 105: 1, 106: 1, 109: 1}

for i in range(0, len(labels)):
    labels[i] = int(labels[i])
    labels[i] = m[labels[i]]

n_output = set(labels)
n_output = len(n_output)

print (sys.getsizeof(features), sys.getsizeof(labels))

fuzzy = FNN(features, labels)
fuzzy.make_model(n_inputs=4, n_hidden1=4, n_hidden2=4, n_outputs=n_output)

