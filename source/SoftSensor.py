from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch import optim
from metric import *
from sklearn.preprocessing import MinMaxScaler


def iterabel_dataset_generation(path):
    df = pd.read_csv(path, header=None)

    # Normalize the data
    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    mask_data = mask_list(shuffled_df.values, 0.5)
    dataset = Dataset.from_pandas(shuffled_df.iloc[:, 1:])
    label = shuffled_df.iloc[:, 0].values.astype(int)
    mask_label = label.copy()
    mask_data = shuffle(mask_data, random_state=1415)
    mask_label[mask_data] = 0
    iterable_dataset = dataset.to_iterable_dataset()
    return iterable_dataset, label, mask_label


def propagate_labels(centers_idx, labels, distances, buffer):

    # 先为没有标签的中心找到最近的、已经标记的样本，并传播标签
    for center in centers_idx:
        if propagated_labels[center] == 2:
            sorted_neighbors = np.argsort(distances[center])
            for neighbor in sorted_neighbors:
                if propagated_labels[neighbor] != 2:
                    propagated_labels[center] = labels[neighbor]
                    break

    # 然后，传播中心的标签到其他未标记的样本中
    for i in range(len(labels)):
        if propagated_labels[i] == 2 & np.isin(i, buffer):
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

def autoencoder_loss(x, x_reconstructed, z, centers_idx, distance_matrix, mask_label):
    reconstruction_loss = nn.MSELoss()(x, x_reconstructed)
    contrastive_loss = contrastive_learning_loss(centers_idx, distance_matrix, mask_label)

    return reconstruction_loss + contrastive_loss

def rejection_model_loss(r, prediction, mask_label, lambda_):
    first_term = 1 + 0.5 * (np.sum(r.detach().numpy()) - np.sum((prediction.detach().numpy() == mask_label) & (prediction.detach().numpy() != 0)))
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
        buffer = np.array(np.where(rejection_value > 0))
        dynamic_distance_matrix.matric_initialization(reduced_instance.detach().numpy())
        distance_matrix = dynamic_distance_matrix.get_distance_matrix()
        centers_idx = dynamic_distance_matrix.density_peak_clustering(distance_matrix)
        propagate_labels(centers_idx, mask_label[:number_of_warning], distance_matrix, buffer)
        loss_ae = autoencoder_loss(warm_data_tensor, reconstructed_instance, reduced_instance, centers_idx, distance_matrix, mask_label[:numberofwarming])
        loss_rm = rejection_model_loss(rejection_value, propagated_labels[:numberofwarming], mask_label[:numberofwarming],  0.1)

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
    rejection_value = rejection_model(reduced_instance)
    buffer = np.array(np.where(rejection_value > 0))

    dynamic_distance_matrix.matrix_update(reduced_instance.detach().numpy())
    distance_matrix = dynamic_distance_matrix.get_distance_matrix()
    centers_idx = dynamic_distance_matrix.density_peak_clustering(distance_matrix)
    propagate_labels(centers_idx, mask_label[index - numberofwarming+ 1: index+1], distance_matrix, buffer)

    if index not in centers_idx:
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
                    break

    loss_ae = autoencoder_loss(new_instance_tensor, reconstructed_instance, reduced_instance, centers_idx, distance_matrix,
                               mask_label[index - numberofwarming+ 1: index +1])
    loss_rm = rejection_model_loss(rejection_value, propagated_labels[index - numberofwarming + 1: index+1], mask_label[index - numberofwarming+ 1: index+1],
                                   0.1)

    # 更新自编码器
    optimizer_ae.zero_grad()
    loss_ae.backward(retain_graph=True)
    optimizer_ae.step()

    # 更新拒绝模型
    optimizer_rm.zero_grad()
    loss_rm.backward()
    optimizer_rm.step()




if __name__ == '__main__':
    path = '../data/musk.csv'
    iterable_dataset, label, mask_label = iterabel_dataset_generation(path)
    ground_truth = []
    number_of_warning = 500

    #model initialization
    dynamic_distance_matrix = DynamicDistanceMatrix(500, 20, 5)
    autoencoder = AutoEncoder(input_dim=166, latent_dim=20)
    rejection_model = RejectionModel(input_dim=20)
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)
    optimizer_rm = optim.Adam(rejection_model.parameters(), lr=0.001)

    propagated_labels = torch.tensor(mask_label, dtype=torch.int64)  # 转换为torch张量

    warm_data_array = np.zeros((number_of_warning, 166))
    for index, example in enumerate(iterable_dataset):
        example = list(example.values())
        instance = np.array(example)
        if index < number_of_warning:
            warm_data_array[index] = instance

            if index == number_of_warning - 1:
                warm(warm_data_array, number_of_warning, 5)
                print('---------------------------------')
        else:
            print(index)
            train(instance,number_of_warning, index)

    print(propagated_labels)
