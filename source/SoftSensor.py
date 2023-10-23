import torch
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch import optim
from metric import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score

def iterabel_dataset_generation(path):
    df = pd.read_csv(path, header=None)

    # Normalize the data
    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    mask_data = mask_list(shuffled_df.values, 0.8)
    dataset = Dataset.from_pandas(shuffled_df.iloc[:, 1:])
    label = shuffled_df.iloc[:, 0].values.astype(int)
    mask_label = label.copy()
    mask_data = shuffle(mask_data, random_state=1415)
    mask_label[mask_data] = 0
    iterable_dataset = dataset.to_iterable_dataset()
    return iterable_dataset, label, mask_label


def propagate_labels(centers_idx, labels, propagated_labels, distances, buffer):
    # 先为没有标签的中心找到最近的、已经标记的样本，并传播标签
    for center in centers_idx:
        if labels[center] == 0:
            sorted_neighbors = np.argsort(distances[center])
            for neighbor in sorted_neighbors:
                if labels[neighbor] != 0:
                    propagated_labels[center] = labels[neighbor]
                    break

    # 然后，传播中心的标签到其他未标记的样本中
    for i in range(len(labels)):
        if np.isin(i, buffer) & ~np.isin(i, centers_idx):
            distances_to_centers = distances[i, centers_idx]
            sorted_center_indices = np.argsort(distances_to_centers)
            propagated_labels[i] = propagated_labels[centers_idx[sorted_center_indices[0]]]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # 编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # 解码
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


# 2. 定义拒绝模型 (Rejection Model)
class RejectionModel(nn.Module):
    def __init__(self, input_dim):
        super(RejectionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CombinedModel, self).__init__()

        # AutoEncoder模型
        self.autoencoder = AutoEncoder(input_dim, latent_dim)

        # RejectionModel模型
        self.rejector = RejectionModel(input_dim)

    def forward(self, x):
        # 通过AutoEncoder
        z, x_reconstructed = self.autoencoder(x)

        # 通过RejectionModel
        reject_val = self.rejector(x)

        return reject_val, x_reconstructed, z



def autoencoder_loss(x, x_reconstructed, z, centers_idx, distance_matrix, mask_label):
    reconstruction_loss = nn.MSELoss()(x, x_reconstructed)
    contrastive_loss = contrastive_learning_loss(centers_idx, distance_matrix, mask_label)
    return reconstruction_loss + contrastive_loss

def rejection_model_loss(r, prediction, mask_label, lambda_):
    first_term = 1 + 0.5 * (np.sum(r.detach().numpy()) - np.sum((prediction.detach().numpy() == mask_label) & (mask_label != 0)))
    secondterm = lambda_ * (1 - (1 / (2 * lambda_ - 1 ) * np.sum(r.detach().numpy())))
    result = np.max([first_term, secondterm, 0])
    loss_tensor = torch.tensor([result], requires_grad=True)
    return loss_tensor


def contrastive_learning_loss(centers, pairdistance, lable):
    loss = 0
    index = np.where(lable != 0)[0]
    closetcenter = []
    for i in (index):
        for center in centers:

            if lable[i] == lable[center] and int(i) < int(center):
                closetcenter.append(pairdistance[i, center])
            elif lable[i] == lable[center] and int(i) > int(center):
                closetcenter.append(pairdistance[center, i])
            else:
                closetcenter.append(0.01)

        loss = loss + min(closetcenter)
        closetcenter = []

    loss_tensor = torch.tensor([loss], requires_grad=True)
    return loss_tensor

def warm(warm_data_array, numberofwarming, epoch):
    warm_data_tensor = torch.tensor(warm_data_array, dtype=torch.float32)  # 转换为torch张量
    for _ in range(epoch):
        reduced_instance, reconstructed_instance = autoencoder(warm_data_tensor)
        rejection_value = rejection_model(reduced_instance)
        #rejection_value, reconstructed_instance, reduced_instance = model(warm_data_tensor)
        buffer = np.array(np.where(rejection_value.squeeze() > 0))[0]
        dynamic_distance_matrix.matric_initialization(reduced_instance.detach().numpy())
        distance_matrix = dynamic_distance_matrix.get_distance_matrix()
        centers_idx = dynamic_distance_matrix.density_peak_clustering(distance_matrix)
        propagate_labels(centers_idx, mask_label[:number_of_warning], propagated_labels[:number_of_warning],distance_matrix, buffer)
        loss_ae = autoencoder_loss(warm_data_tensor, reconstructed_instance, reduced_instance, centers_idx, distance_matrix, mask_label[:numberofwarming])
        loss_rm = rejection_model_loss(rejection_value, propagated_labels[:numberofwarming], mask_label[:numberofwarming],5)

        #loss = loss_ae + loss_rm
        # 更新自编码器
        optimizer_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        optimizer_ae.step()

        # 更新拒绝模型
        optimizer_rm.zero_grad()
        loss_rm.backward()
        optimizer_rm.step()

def train(new_instance, numberofwarming, index):
    new_instance_tensor = torch.tensor(new_instance, dtype=torch.float32)  # 转换为torch张量
    reduced_instance, reconstructed_instance = autoencoder(new_instance_tensor)

    #_, _, reduced_instance = model(new_instance_tensor)
    dynamic_distance_matrix.matrix_update(reduced_instance.detach().numpy())
    distance_matrix = dynamic_distance_matrix.get_distance_matrix()
    centers_idx = dynamic_distance_matrix.density_peak_clustering(distance_matrix)
    rejection_value = rejection_model(torch.Tensor(dynamic_distance_matrix.data))

    buffer = np.array(np.where(rejection_value.squeeze() > 0))[0]

    propagate_labels(centers_idx, mask_label[index - numberofwarming+ 1: index+1], propagated_labels[index - numberofwarming + 1: index+1], distance_matrix, buffer)

    loss_ae = autoencoder_loss(new_instance_tensor, reconstructed_instance, reduced_instance, centers_idx, distance_matrix,
                               mask_label[index - numberofwarming+ 1: index +1])
    loss_rm = rejection_model_loss(rejection_value, propagated_labels[index - numberofwarming + 1: index+1], mask_label[index - numberofwarming+ 1: index+1],
                                   5)

    # 更新自编码器
    optimizer_ae.zero_grad()
    loss_ae.backward(retain_graph=True)
    optimizer_ae.step()

    # 更新拒绝模型
    optimizer_rm.zero_grad()
    loss_rm.backward()
    optimizer_rm.step()

def compute_accuracy(labels, predictions):
    assert len(labels) == len(predictions), "标签和预测值数量必须相同"

    correct_count = 0
    total_count = 0

    for label, pred in zip(labels, predictions):
        if pred == 0:
            continue
        if label == pred:
            correct_count += 1
        total_count += 1

    if total_count == 0:
        return 0  # 避免除以零的情况
    return correct_count / total_count
def compute_f1_transformed(labels, predictions):
    assert len(labels) == len(predictions), "标签和预测值数量必须相同"

    # 把-1和1都当作1来处理
    labels_transformed =np.abs(labels)
    predictions_transformed = np.abs(predictions)
    print(recall_score(labels_transformed, predictions_transformed, pos_label=0))
    print(precision_score(labels_transformed, predictions_transformed, pos_label=0))
    return f1_score(labels_transformed, predictions_transformed, pos_label=0)

if __name__ == '__main__':
    path = '../data/musk.csv'
    iterable_dataset, label, mask_label = iterabel_dataset_generation(path)
    ground_truth = []
    number_of_warning = 200

    #model initialization
    dynamic_distance_matrix = DynamicDistanceMatrix(number_of_warning, 20, 10)
    autoencoder = AutoEncoder(input_dim=166, latent_dim=20)
    rejection_model = RejectionModel(input_dim=20)
    #model = CombinedModel(166, 20)
    optimizer_ae = optim.SGD(autoencoder.parameters(), lr=0.001)
    optimizer_rm = optim.SGD(rejection_model.parameters(), lr=0.001)

    propagated_labels = torch.tensor(mask_label, dtype=torch.int64)  # 转换为torch张量

    warm_data_array = np.zeros((number_of_warning, 166))
    for index, example in enumerate(iterable_dataset):
        example = list(example.values())
        instance = np.array(example)
        if index < number_of_warning:
            warm_data_array[index] = instance

            if index == number_of_warning - 1:
                warm(warm_data_array, number_of_warning, 15)
                print('---------------------------------')
        else:
            train(instance,number_of_warning, index)

    print(np.sum(label == 0))
    print(np.sum(mask_label == 0))
    print(np.sum(propagated_labels.numpy() == 0))
    acc = compute_accuracy(label, propagated_labels)
    f1 = compute_f1_transformed(label, propagated_labels)

    print(acc)
    print(f1)

'''   if index not in centers_idx:
    if propagated_labels[index] == 0:
        distances_to_centers = distance_matrix[-1, centers_idx]
        sorted_center_indices = np.argsort(distances_to_centers)
        propagated_labels[index] = propagated_labels[centers_idx[sorted_center_indices[0]]]
else:
    if propagated_labels[index] == 0:
        sorted_neighbors = np.argsort(distance_matrix[index])
        for neighbor in sorted_neighbors:
            if propagated_labels[neighbor] != 0:
                propagated_labels[index] = propagated_labels[neighbor]
                break'''