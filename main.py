# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:07:13 2021

@author: Ahmed
"""
import mne
from Read_Data import Read_Data
from Feature_Extraction import Feature_Extraction
from Preprocessing import Preprocessing

# use gpu cuda cores
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

edf_01_tcp_ar = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

raw, name = edf_01_tcp_ar.data[200]

print (edf_01_tcp_ar.labels.head())
print (edf_01_tcp_ar.labels.columns)
print (edf_01_tcp_ar.labels[[4]].describe())

print (raw.ch_names)

# channels to work on: FP1, FP2

preprocessing = Preprocessing()
raw = preprocessing.select_ch(raw)
raw.plot()
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
