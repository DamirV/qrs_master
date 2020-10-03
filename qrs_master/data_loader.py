import json
import random
from torch.utils.data import Dataset, DataLoader


class QrsDataset(Dataset):
    def __init__(self):
        self.data = self.load_data()
        self.positiveCount = 0
        self.negativeCount = 0
        self.isQRS = False

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
        randomFlag = 0 #random.randint(0, 1)

        if(randomFlag == 1):
            self.positiveCount += 1
            self.isQRS = True
            center = random.choice(deliniation)[1]
        else:
            self.negativeCount += 1
            self.isQRS = False
            center = self.randCenter(deliniation, signal)

        signal = signal[center - 30:center + 31]
        return signal


    def __len__(self):
        return 200 # заглушка


    def randCenter(self, deliniation, signal):
        center = random.randint(0 + 30, 5000 - 31)

        for i in deliniation:
            intersection = (center > (i[1] - 60)) and (center < (i[1] + 61))
            if(intersection):
                center = self.randCenter(deliniation, signal)
                break

        return center


    def IsQRS(self):
        return self.isQRS