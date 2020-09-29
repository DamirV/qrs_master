import json


def load_data():
    PATH = "D:\\Projects\\ecg_gan_experiments-master\\Dataset\\"
    FILENAME = "ecg_data_200.json"
    json_file = PATH + FILENAME
    with open(json_file, 'r') as f:
        data = json.load(f)

    return data
