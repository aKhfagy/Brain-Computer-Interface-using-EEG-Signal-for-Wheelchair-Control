# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:27:58 2021

@author: Ahmed
"""
import mne
import pandas as pd

class EDF_Data:
    def __init__(self, data, labels, montage, labels_map):
        self.data = data
        self.labels = labels
        self.montage = montage
        self.labels_map = labels_map

class Read_Data:
    def __init__(self, path_edf, path_labels, path_montage, path_labels_map):
        self.path_edf = path_edf
        self.path_labels = path_labels
        self.path_montage = path_montage
        self.path_labels_map = path_labels_map
        
    def get_data(self):
        data = []
        labels = []
        montage = []
        labels_map = []
        # read list from path path_file_list
        paths_file = open(self.path_edf, 'r')
        paths_edf = paths_file.readlines()
        print ("File Paths: ")
        for path in paths_edf:
            striped_path = path.strip()
            striped_path = striped_path[2:]
            striped_path = "TUAR/v2.0.0" + striped_path
            name = striped_path[-22:]
            name = name[:-4]
            # read data from the paths and add it to data object
            raw = mne.io.read_raw_edf(striped_path)
            data.append((raw, name))
            
        del paths_file, paths_edf
        # read labels
        labels = pd.read_csv(self.path_labels, skiprows=7, header=None)
        # read montage (contains the full name of the channel from the accronim in the labels file)
        
        # read labels map
        
        
        ret_value = EDF_Data(data, labels, montage, labels_map)
        return ret_value
    
# testing
print ("testing...")
read_data = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt",
                      "TUAR/v2.0.0/_DOCS/annotation_label_map.txt")

data = read_data.get_data()
print (data.data)
print (data.labels.head(2))
print (data.labels_map)
print (data.montage)
del read_data