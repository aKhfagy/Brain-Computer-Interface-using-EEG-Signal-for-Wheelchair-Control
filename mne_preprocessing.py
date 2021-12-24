# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:44:23 2021

@author: Ahmed
"""

# import packages
import mne

mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

file = "TUAR/v2.0.0/edf/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.edf"
raw = mne.io.read_raw_edf(file)
print (raw)
print (raw.info)
print (raw.ch_names)
raw.plot_psd(fmax=50)
raw.plot(duration=5, n_channels=36)

print("end")