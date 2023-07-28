import pandas as pd
from sklearn import *
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer
from metric import *




def musk_normalization(random_seed, label_rate):
    path = '../data/musk.csv'
    df = pd.read_csv(path, header=None)
    dataset = df.values
    data = dataset[:, 1:]
    label = dataset[:,0 ].astype(int)
    scaler = Normalizer().fit(data)
    data_norm = scaler.transform(data)
    mask_data = mask_list(torch.Tensor(data_norm), label_rate)
    mask_label = label.copy()
    mask_data = shuffle(mask_data, random_state=1415)
    mask_label[mask_data] = 0
    data, label, mask_label = shuffle(data_norm, label, mask_label, random_state=random_seed)
    for i in range(10):
        data[int(2196* (0.1 * (i))):int(2196 * (0.1 * (i + 1))), int(166 * (0.1 * (i + 1))):] = 0

    return data, label, mask_label