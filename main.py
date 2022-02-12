# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:07:13 2021

@author: Ahmed
"""
import os
import sys

import mne
from Read_Data import Read_Data
from Feature_Extraction import Feature_Extraction
from Preprocessing import Preprocessing

# use gpu cuda cores
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# channels to work on: FP1, FP2

edf_01_tcp_ar = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

preprocessing = Preprocessing()

raw_ch = []
labels = []

for data in edf_01_tcp_ar.data:
    raw, name = data
    raw = preprocessing.select_ch(raw)
    label = edf_01_tcp_ar.labels[edf_01_tcp_ar.labels[0] == name]
    raw_ch.append(raw)
    labels.append(label)

del edf_01_tcp_ar

# 'EEG FP1-LE', 'EEG FP2-LE'

edf_02_tcp_le = Read_Data("TUAR/v2.0.0/lists/edf_02_tcp_le.list",
                      "TUAR/v2.0.0/csv/labels_02_tcp_le.csv",
                      "TUAR/v2.0.0/_DOCS/02_tcp_le_montage.txt").get_data()

for data in edf_02_tcp_le.data:
    raw, name = data
    raw = preprocessing.select_ch(raw, ch_names=['EEG FP1-LE', 'EEG FP2-LE'])
    label = edf_02_tcp_le.labels[edf_02_tcp_le.labels[0] == name]
    raw_ch.append(raw)
    labels.append(label)

del edf_02_tcp_le

edf_03_tcp_ar_a = Read_Data("TUAR/v2.0.0/lists/edf_03_tcp_ar_a.list",
                      "TUAR/v2.0.0/csv/labels_03_tcp_ar_a.csv",
                      "TUAR/v2.0.0/_DOCS/03_tcp_ar_a_montage.txt").get_data()

for data in edf_03_tcp_ar_a.data:
    raw, name = data
    raw = preprocessing.select_ch(raw)
    label = edf_03_tcp_ar_a.labels[edf_03_tcp_ar_a.labels[0] == name]
    raw_ch.append(raw)
    labels.append(label)

del edf_03_tcp_ar_a

print (sys.getsizeof(raw_ch), sys.getsizeof(labels))

print (raw_ch)
print (labels)

# print (preprocessing.get_time_range_arr(raw, 10, 30))
# print (preprocessing.get_time_range_raw(raw, 10, 30))

# print (preprocessing.get_shape(raw))

# epochs = preprocessing.get_epochs(raw, raw.ch_names)

# print (preprocessing.scale(raw, epochs))

# print (preprocessing.denoise(raw, 1))

# feature_extraction = Feature_Extraction()

# print (feature_extraction.CSP(raw, name, 4))

#print (feature_extraction.FFT(raw, 4))

# del edf_01_tcp_ar

# edf_02_tcp_le = Read_Data("TUAR/v2.0.0/lists/edf_02_tcp_le.list", 
#                       "TUAR/v2.0.0/csv/labels_02_tcp_le.csv",
#                       "TUAR/v2.0.0/_DOCS/02_tcp_le_montage.txt").get_data()

# del edf_02_tcp_le

# edf_03_tcp_ar_a = Read_Data("TUAR/v2.0.0/lists/edf_03_tcp_ar_a.list", 
#                       "TUAR/v2.0.0/csv/labels_03_tcp_ar_a.csv",
#                       "TUAR/v2.0.0/_DOCS/03_tcp_ar_a_montage.txt").get_data()


# del edf_03_tcp_ar_a
