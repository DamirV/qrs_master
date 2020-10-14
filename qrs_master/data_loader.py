import json
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class QrsDataset(Dataset):
    def __init__(self):
        self.data = self.load_data()


    def load_data(self):
        PATH = "D:\\Projects\\ecg_gan_experiments-master\\Dataset\\"
        PATH2 = "/home/a/PycharmProjects/qrs_master/dataset/"
        FILENAME = "ecg_data_200.json"
        json_file = PATH + FILENAME
        with open(json_file, 'r') as f:
            data = json.load(f)

        return data


    def __getitem__(self, id):
        key, value = random.choice(list(self.data.items()))
        signal = value['Leads']['i']['Signal']
        deliniation = value['Leads']['i']['Delineation']['qrs']
        randomFlag = random.randint(0, 1)

        if(randomFlag == 1):
            isqrs = 1
            center = random.choice(deliniation)[1]
        else:
            isqrs = 0
            center = self.randCenter(deliniation, signal)

        signal = signal[center - 30:center + 31]

        signal = np.asarray(signal, dtype=np.float32)
        isqrs = np.asarray(isqrs, dtype=np.float32)

        signal = torch.from_numpy(signal)
        isqrs = torch.from_numpy(isqrs)

        return signal, isqrs


    def __len__(self):
        return 2000 # заглушка


    def randCenter(self, deliniation, signal):
        center = random.randint(0 + 30, 5000 - 31)

        for i in deliniation:
            intersection = (center > (i[1] - 60)) and (center < (i[1] + 61))
            if(intersection):
                center = self.randCenter(deliniation, signal)
                break

        return center


