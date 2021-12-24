# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:27:58 2021

@author: Ahmed
"""
import mne

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
            # read data from the paths and add it to data object
            raw = mne.io.read_raw_edf(striped_path)
            data.append(raw)
            
        del paths_file, paths_edf
        # read labels
        
        # read montage (contains the full name of the channel from the accronim in the labels file)
        
        # read labels map
        
        return (data, labels, montage, labels_map)
    
# testing
print ("testing...")
read_data = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
                      "TUAR/v2.0.0/lists/csv/labels_01_tcp_ar.csv",
                      "TUAR/v2.0.0/lists/_DOCS/01_tcp_ar_montage.txt",
                      "TUAR/v2.0.0/lists/_DOCS/annotation_label_map.txt")

data, labels, montage, labels_map = read_data.get_data()

del read_data, data, labels, montage, labels_map