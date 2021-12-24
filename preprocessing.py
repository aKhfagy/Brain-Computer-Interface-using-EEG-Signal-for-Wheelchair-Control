# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:44:23 2021

@author: Ahmed
"""

# import packages
import mne
from Read_Data import Read_Data

# use gpu cuda cores
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

class Preprocessing:
    def __init__(self, edf_data):
        self.edf_data = edf_data


edf_01_tcp_ar = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

print (edf_01_tcp_ar.data)
print (edf_01_tcp_ar.labels.head(3))
print (edf_01_tcp_ar.montage)
print (edf_01_tcp_ar.LABELS_MAP)

del edf_01_tcp_ar

edf_02_tcp_le = Read_Data("TUAR/v2.0.0/lists/edf_02_tcp_le.list", 
                      "TUAR/v2.0.0/csv/labels_02_tcp_le.csv",
                      "TUAR/v2.0.0/_DOCS/02_tcp_le_montage.txt").get_data()

print (edf_02_tcp_le.data)
print (edf_02_tcp_le.labels.head(3))
print (edf_02_tcp_le.montage)
print (edf_02_tcp_le.LABELS_MAP)

del edf_02_tcp_le

edf_03_tcp_ar_a = Read_Data("TUAR/v2.0.0/lists/edf_03_tcp_ar_a.list", 
                      "TUAR/v2.0.0/csv/labels_03_tcp_ar_a.csv",
                      "TUAR/v2.0.0/_DOCS/03_tcp_ar_a_montage.txt").get_data()

print (edf_03_tcp_ar_a.data)
print (edf_03_tcp_ar_a.labels.head(3))
print (edf_03_tcp_ar_a.montage)
print (edf_03_tcp_ar_a.LABELS_MAP)

del edf_03_tcp_ar_a

