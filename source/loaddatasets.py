import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer
from metric import *
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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



def generated_fail_dataset(path):
    # 1. 读取csv文件
    df = pd.read_csv(path)

    # 2. 选择所有"Labelx-fail"列
    fail_columns = [f'Label{i}-fail' for i in range(11)]
    pass_columns = [f'Label{i}-pass' for i in range(11)]
    # 3. 初始化一个空的DataFrame来存储结果
    result_df = pd.DataFrame()

    # 4. 为每个Labelx-fail列生成新样本
    for column in fail_columns:
        # 选出在特定的Labelx-fail列中值为1的样本
        temp_df = df[df[column] == 1].copy()
        # 创建新的列'new_label'并赋予对应的x值
        temp_df['new_label'] = int(column.split('-')[0].replace('Label', ''))
        # 将这个临时DataFrame添加到结果DataFrame中
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 5. 删除原始的Labelx-fail列
    result_df = result_df.drop(columns=fail_columns)
    result_df = result_df.drop(columns=pass_columns)

    # 6. 保存结果到新的csv文件
    result_df.to_csv('../data/generated_negtive_data.csv', index=False)
    print(result_df)

    # 7. 计算每个不同的new_label值对应的样本数量
    label_counts = result_df['new_label'].value_counts()

    # 8. 打印结果
    print(label_counts)

def oversample(path):
    result_df = pd.read_csv(path)
    # 假设 result_df 是您的数据
    X = result_df.drop(columns='new_label')
    y = result_df['new_label']

    # 对于传感器数据，通常先进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sampling_strat = {0:200,1:200, 2:200,3:200,4:200,5: 5000,6: 5000, 7:200,8:200,9:200,10:200}
    smote = SMOTE(sampling_strategy= sampling_strat)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['new_label'] = y_resampled
    # 然后您可以将X_resampled和y_resampled转换回DataFrame，并保存或进一步处理
    df_resampled.to_csv('../data/generated_negtive_oversample_data.csv', index=False)

path = '../data/generated_negtive_data.csv'
oversample(path)
