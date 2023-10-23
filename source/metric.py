import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import pairwise_distances


def iterabel_dataset_generation(path):
    path = '../data/musk.csv'
    df = pd.read_csv(path, header=None)
    dataset = Dataset.from_pandas(df)
    iterable_dataset = dataset.to_iterable_dataset()
    #    print(np.array(list(example.values())))
    return iterable_dataset

def mask_list(data, ratio):
    data_idx = np.arange(0, len(data))
    train_mask = np.zeros(len(data), dtype=bool)
    train_idx = data_idx[:int(len(data) * ratio)]
    train_mask[train_idx] = True
    return train_mask

class DynamicDistanceMatrix:
    def __init__(self, number_of_instances, number_of_features, n_clusters):
        self.N = number_of_instances
        self.M = number_of_features
        self.centers =  n_clusters
        self.distance_matrix = np.zeros((number_of_instances, number_of_instances))
        self.current_size = 0

    def matric_initialization(self, new_samples):
        # 如果矩阵尚未被填满
            self.data = new_samples
            self.distance_matrix = pairwise_distances(self.data, self.data)
        # 当矩阵被填满后

    def matrix_update(self, new_sample):
        # 在data中移除第一个样本，并在尾部添加新的样本
        self.data[:-1] = self.data[1:]
        self.data[-1] = new_sample

        # 使用numpy的slicing更新距离矩阵
        self.distance_matrix[:-1, :] = self.distance_matrix[1:, :]
        self.distance_matrix[:, :-1] = self.distance_matrix[:, 1:]

        # 使用pairwise_distances计算新样本与所有样本的距离，并更新到矩阵中
        distances = pairwise_distances([new_sample], self.data).flatten()
        self.distance_matrix[-1, :] = distances
        self.distance_matrix[:, -1] = distances

    def density_peak_clustering(self, distance_matrix, dc_percentile=2.0):
        # 计算截断距离dc
        dc = np.percentile(distance_matrix, dc_percentile)
        # 计算每个点的密度
        rho = np.sum(np.exp(-(distance_matrix / dc) ** 2), axis=1) - 1
        # 计算每个点的delta
        max_distance = np.max(distance_matrix)
        delta = np.full_like(rho, max_distance)
        nearest_higher_density = np.full_like(rho, -1, dtype=int)
        for i in range(len(rho)):
            mask = rho > rho[i]
            if mask.any():
                delta[i] = np.min(distance_matrix[i, mask])
                nearest_higher_density[i] = np.argmin(distance_matrix[i, mask])
        # 选择聚类中心
        cluster_centers = np.argsort(delta)[-self.centers:]
        return cluster_centers

    def get_distance_matrix(self):
        return self.distance_matrix


import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


def distanceloss(centers, pairdistance, lable):
    loss = 0

    # 创建一个字典，以便能够根据两个样本的索引快速查找它们之间的距离
    distance_dict = {}
    for i, j, dist in pairdistance:
        distance_dict[(i, j)] = dist
        distance_dict[(j, i)] = dist  # 假设距离是对称的

    labeled_samples = [i for i, lbl in enumerate(lable) if lbl != 0]

    for i in labeled_samples:
        closetcenter = float('inf')  # 使用正无穷大作为初始化值
        for center in centers:
            if lable[i] == lable[center]:
                dist = distance_dict.get((i, center), 1)  # 如果找不到距离，则默认为1
                closetcenter = min(closetcenter, dist)
        loss += closetcenter

    loss_tensor = torch.tensor([loss], requires_grad=True)
    return loss_tensor



