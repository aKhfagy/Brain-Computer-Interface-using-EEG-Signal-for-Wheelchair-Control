# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:09:54 2021

@author: Ahmed
"""

import mne
import pywt

class Feature_Extraction:
    def __init__(self):
        return
    
    def FFT(self, raw, window_size):
        stft = mne.time_frequency.stft(raw.get_data(), wsize=window_size)
        return stft
    
    def wavelet(self, raw):
        ret = pywt.wavedec(raw, wavelet='db4', level=8)
        return ret
    
    def CSP(self, raw, name, n_components, 
            reg=None, log=True, norm_trace=False):
        csp = mne.decoding.CSP(n_components=n_components,
                               reg=reg, log=log,
                               norm_trace=norm_trace)
        picks = mne.pick_types(raw.info, meg=False, 
                               eeg=True, stim=False, 
                               eog=False, exclude='bads')
        events = mne.find_events(raw, raw.ch_names)
        epochs = mne.Epochs(raw, events, 
                            event_repeated='drop', preload=True,
                            picks=picks)
        labels = epochs.events[:, -1] - 2
        csp = csp.fit(epochs.get_data(), labels)
        X_new = csp.transform(epochs.get_data())
        return X_new
    