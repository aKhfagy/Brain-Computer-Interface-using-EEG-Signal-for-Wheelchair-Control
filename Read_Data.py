# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:27:58 2021

@author: Ahmed
"""
import mne
import pandas as pd

class EDF_Data:
    LABELS_MAP = { 0:'null',
                    1:'spsw',
                    2:'gped',
                    3:'pled',
                    4:'eyeb',
                    5:'artf',
                    6:'bckg',
                    7:'seiz',
                    8:'fnsz',
                    9:'gnsz',
                    10:'spsz',
                    11:'cpsz',
                    12:'absz',
                    13:'tnsz',
                    14:'cnsz',
                    15:'tcsz',
                    16:'atsz',
                    17:'mysz',
                    18:'nesz',
                    19:'intr',
                    20:'slow',
                    21:'eyem',
                    22:'chew',
                    23:'shiv',
                    24:'musc',
                    25:'elpp',
                    26:'elst',
                    27:'calb',
                    28:'hphs',
                    29:'trip',
                    30:'elec',
                    100:'eyem_chew',
                    101:'eyem_shiv',
                    102:'eyem_musc',
                    103:'eyem_elec',
                    104:'chew_shiv',
                    105:'chew_musc',
                    106:'chew_elec',
                    107:'shiv_musc',
                    108:'shiv_elec',
                    109:'musc_elec'}
    def __init__(self, data, labels, montage):
        self.data = data
        self.labels = labels
        self.montage = montage


class Read_Data:
    
    def __init__(self, path_edf, path_labels, path_montage):
        self.path_edf = path_edf
        self.path_labels = path_labels
        self.path_montage = path_montage
        
    def get_data(self):
        data = []
        labels = []
        montage = {}
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
        montage_file = open(self.path_montage, 'r')
        montage_file_lines = montage_file.readlines()
        
        for line in montage_file_lines:
            if(line[0] == 'm'):
                temp = line.split(', ')
                key_value = temp[1].split(': ')
                values = key_value[1].split(' -- ')
                key = key_value[0]
                montage[key] = values
        
        del montage_file

        ret_value = EDF_Data(data, labels, montage)
        return ret_value
    
# testing
# print ("testing...")
# read_data = Read_Data("TUAR/v2.0.0/lists/edf_01_tcp_ar.list", 
#                       "TUAR/v2.0.0/csv/labels_01_tcp_ar.csv",
#                       "TUAR/v2.0.0/_DOCS/01_tcp_ar_montage.txt")

# data = read_data.get_data()
# print (data.data)
# print (data.labels.head(2))
# print (data.montage)
# print (data.LABELS_MAP)
# del read_data, data