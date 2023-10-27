import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer
from metric import *
import torch


def mask_list(data, ratio):
    data_idx = torch.arange(0, data.size(0))
    train_mask = torch.zeros(data.size(0), dtype=torch.bool)
    train_idx = data_idx[:int(data.size(0) * ratio)]
    train_mask[train_idx] = True
    return train_mask

def musk_normalization(random_seed, label_rate):
    path = '../data/musk.csv'
    df = pd.read_csv(path, header=None)
    dataset = df.values
    data = dataset[:, 1:]
    label = dataset[:,0 ].astype(int)
    scaler = Normalizer().fit(data)
    data_norm = scaler.transform(data)
    mask_data = mask_list(torch.Tensor(data_norm), label_rate)
    print(mask_data)
    mask_label = label.copy()
    mask_data = shuffle(mask_data, random_state=1415)
    mask_label[mask_data] = 0
    data, label, mask_label = shuffle(data_norm, label, mask_label, random_state=random_seed)
    for i in range(10):
        data[int(2196* (0.1 * (i))):int(2196 * (0.1 * (i + 1))), int(166 * (0.1 * (i + 1))):] = 0

    return data, label, mask_label

import pandas as pd
def iterabel_dataset_generation(path):
    df = pd.read_csv(path)
    labels_pass = [f"Label{i}-pass" for i in range(11)]
    labels_fail = [f"Label{i}-fail" for i in range(11)]
    # 初始化新的label列为0
    df["label"] = 12

    # 先根据每个Fail列进行更新
    for i, label in enumerate(labels_fail):
        df.loc[df[label] == 1, "label"] = i + 1
    # 仅当new_label为0时，考虑Pass列
    df.loc[(df[labels_pass].sum(axis=1) == 1) & (df["label"] == 12), "label"] = 0
    # 结果展示，这里我假设你的sensor列是这样命名的sensor0, sensor1,..., sensor93
    sensors = [f"Sensor{i+1}" for i in range(94)]
    df = df[sensors + ["label"] + labels_pass + labels_fail]



    # 计算new_label列中值不为0的样本数量
    non_zero_count = ((df['label'] != 0) & (df['label'] != 12)).sum()

    # 从值为0的样本中随机采样
    #sampled_zero_df = df[df['label'] == 0 ].sample(n=1 * non_zero_count, replace=False)
    sampled_zero_df = df[(df['label'] == 0) & (df['Label10-pass'] == 1)]
    print(sampled_zero_df)
    sampled_zero_df = sampled_zero_df.sample(n=1 * non_zero_count, replace=False)

    # 获取值不为0的样本
    non_zero_df = df[(df['label'] != 0) & (df['label'] != 12)]


    # 将两部分数据组合
    final_df = pd.concat([sampled_zero_df, non_zero_df], axis=0)

    label_columns = ['Label0-pass', 'Label0-fail', 'Label1-pass', 'Label1-fail', 'Label2-pass', 'Label2-fail',
                     'Label2-pass', 'Label2-fail'
        , 'Label3-pass', 'Label3-fail', 'Label4-pass', 'Label4-fail', 'Label5-pass', 'Label5-fail', 'Label6-pass',
                     'Label6-fail'
        , 'Label7-pass', 'Label7-fail', 'Label8-pass', 'Label8-fail', 'Label9-pass', 'Label9-fail', 'Label10-pass',
                     'Label10-fail']
    final_df = final_df.drop(columns=label_columns)


    # 保存为CSV文件
    final_df.to_csv('../data/generated_final_dataset.csv', index=False)
    return final_df.values[:, :-1], final_df.values[:, -1]


path = '../data/final.csv'
data, label = iterabel_dataset_generation(path)
