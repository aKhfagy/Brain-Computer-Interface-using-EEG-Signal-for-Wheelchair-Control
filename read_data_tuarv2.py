import mne
import pandas as pd

class EDFDataTUARv2:
    LABELS_MAP_NUMBER_NAME = { 0:'null',
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
    
    LABELS_MAP_NAME_NUMBER = { 'null':0,
                    'spsw':1,
                    'gped':2,
                    'pled':3,
                    'eyeb':4,
                    'artf':5,
                    'bckg':6,
                    'seiz':7,
                    'fnsz':8,
                    'gnsz':9,
                    'spsz':10,
                    'cpsz':11,
                    'absz':12,
                    'tnsz':13,
                    'cnsz':14,
                    'tcsz':15,
                    'atsz':16,
                    'mysz':17,
                    'nesz':18,
                    'intr':19,
                    'slow':20,
                    'eyem':21,
                    'chew':22,
                    'shiv':23,
                    'musc':14,
                    'elpp':25,
                    'elst':26,
                    'calb':27,
                    'hphs':28,
                    'trip':29,
                    'elec':30,
                    'eyem_chew':100,
                    'eyem_shiv':101,
                    'eyem_musc':102,
                    'eyem_elec':103,
                    'chew_shiv':104,
                    'chew_musc':105,
                    'chew_elec':106,
                    'shiv_musc':107,
                    'shiv_elec':108,
                    'musc_elec':109}
    def __init__(self, data, labels, montage):
        self.data = data
        self.labels = labels
        self.montage = montage


class ReadDataTUARv2:
    
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
            raw = mne.io.read_raw_edf(striped_path, preload=True)
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

        ret_value = EDFDataTUARv2(data, labels, montage)
        return ret_value
    