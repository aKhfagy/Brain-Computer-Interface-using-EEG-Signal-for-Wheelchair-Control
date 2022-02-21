from os import walk
from scipy.io import loadmat


class ReadDataMotorImaginary:
    def __init__(self):
        filenames = next(walk('motor_dataset/'), (None, None, []))[2]
        self.files = []
        for i in range(len(filenames)):
            print (i + 1, '/', len(filenames))
            file = loadmat('motor_dataset/' + filenames[i])
            self.files.append(file)
        return
    def get_data(self):
        return self.files

