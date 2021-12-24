# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:44:23 2021

@author: Ahmed
"""

# import packages
import mne
import numpy as np
from Read_Data import Read_Data

class Preprocessing:
    def __init__(self):
        return
    
    def get_time_range_arr(self, raw, start_time, end_time, channel_names = None):
        sampling_freq = raw.info['sfreq']
        start_stop_seconds = np.array([start_time, end_time])
        start, end = (start_stop_seconds * sampling_freq).astype(int)
        selection = None
        if channel_names is None:
            selection = raw[:, start:end]
        else:
            selection = raw[channel_names, start:end]
        return selection
    
    def get_shape(self, raw):
        return raw.get_data().shape
    
    def get_time_range_raw(self, raw, start_time, end_time):
        raw_select = raw.copy().crop(tmin=start_time, tmax=end_time)
        return raw_select
    
    def get_epochs(self, raw, channels, tmin=-0.2, tmax=0.5):
        events = mne.find_events(raw, channels)
        epochs = mne.Epochs(raw, events, 
                            tmin=tmin, tmax=tmax, 
                            event_repeated='drop')
        return epochs
    
    def scale(self, raw, epochs):
        scaler = mne.decoding.Scaler(epochs.info)
        epochs_data = epochs.get_data()
        scaler.fit(epochs_data)
        scaled_data = scaler.transform(epochs_data)
        return scaled_data
    
    def denoise(self, raw, number_components):
        picks = mne.pick_types(raw.info, meg=False, 
                               eeg=True, stim=False, 
                               eog=False, exclude='bads')
        
        events = mne.find_events(raw, raw.ch_names)
        epochs = mne.Epochs(raw, events, 
                            event_repeated='drop', preload=True,
                            picks=picks)
        
        signal_cov = mne.compute_raw_covariance(raw, picks=picks)
        
        xd = mne.preprocessing.Xdawn(n_components=number_components, 
                                     signal_cov=signal_cov)
        
        xd.fit(epochs)
        epochs_denoised = xd.apply(epochs)
        
        return epochs_denoised
    
    def filter(self, raw, low_freq=None, high_freq=None):
        filtered_raw = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
        return filtered_raw


