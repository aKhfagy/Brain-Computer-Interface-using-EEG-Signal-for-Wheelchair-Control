from os import walk
from scipy.io import loadmat


class ReadDataMotorImaginary:
    def __init__(self):
        self.filenames = next(walk('motor_dataset/'), (None, None, []))[2]
        self.files = []
        for i in range(len(self.filenames)):
            print (i + 1, '/', len(self.filenames))
            file = loadmat('motor_dataset/' + self.filenames[i])
            self.files.append(file)
        return
    def get_data(self):
        return self.files, self.filenames

