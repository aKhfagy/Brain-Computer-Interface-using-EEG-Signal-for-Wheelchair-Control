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

edf_01_tcp_ar = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt").get_data()

print (edf_01_tcp_ar.data)
print (edf_01_tcp_ar.labels.head(3))
print (edf_01_tcp_ar.montage)
print (edf_01_tcp_ar.LABELS_MAP)

