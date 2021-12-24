# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:09:54 2021

@author: Ahmed
"""

import mne

class Feature_Extraction:
    def __init__(self):
        return
    
    def FFT(self, raw, window_size):
        stft = mne.time_frequency.stft(raw.get_data(), wsize=window_size)
        return stft
    
    def wavelet(self, raw):
        return
    
    def CSP(self, raw):
        
        return
    