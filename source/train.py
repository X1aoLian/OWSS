import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

def dataset_generation(random_seed):
    # 设置随机种子以便结果可复现
    np.random.seed(random_seed)

    n_samples = 10000  # 每组样本数
    n_features = 200  # 样本特征数
    # 生成数据
    cluster1 = np.random.normal(np.zeros(n_features), 1, size=(n_samples, n_features))
    cluster2 = np.random.normal(np.ones(n_features) * 3, 1, size=(n_samples, n_features))
    cluster3 = np.random.normal(np.ones(n_features) * 100, 1, size=(5000, n_features))

    # 分配标签
    labels1 = np.zeros(n_samples)
    labels2 = np.ones(n_samples)
    labels3 = np.ones(5000)

    # 组合数据和标签
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = np.hstack([labels1, labels2, labels3])

    # 生成index
    index = np.array([0] * n_samples + [1] * n_samples + [2] * 5000)

    # Shuffle数据，标签和index
    data, labels, index = shuffle(data, labels, index)

    return data, labels, index


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 200),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Rejection model
        self.rejection = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        cls_out = self.classifier(encoded)
        rej_out = self.rejection(encoded)
        return decoded, cls_out, rej_out

def rejection_loss(rejection_value, loss, D ,lambda_):

    first_term = rejection_value + loss - D
    secondterm = lambda_ * (1 - (1 / (2 * lambda_ - 1) * rejection_value))

    result = torch.max(torch.max(first_term, secondterm), torch.tensor(0).to(first_term.device))

    #loss_tensor = torch.tensor([result], requires_grad=True)
    return result


def evaluate_metrics(true_labels, predicted_labels):
    # 1. Calculate accuracy for predicted values of 0 or 1
    mask = predicted_labels != 2
    accuracy = (true_labels[mask] == predicted_labels[mask]).float().mean().item()

    # 2.
    true_binary = (true_labels == 2).numpy()
    predicted_binary = (predicted_labels == 2).numpy()

    f1 = f1_score(true_binary, predicted_binary)
    recall = recall_score(true_binary, predicted_binary)
    precision = precision_score(true_binary, predicted_binary)

    print(f"Accuracy (only for predicted 0 or 1): {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel().to(device)
criterion_ae = nn.MSELoss()
criterion_cls = nn.BCELoss()
criterion_rej = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

data, labels, index = dataset_generation(42)
data, labels, index = torch.tensor(data), torch.tensor(labels), torch.tensor(index)
prediction = []

for epoch in range(1):
    for i, x in enumerate(data):

        input = x.to(device).float()
        label = labels[i].to(device).float().unsqueeze(0)

        optimizer.zero_grad()
        decoded, cls_out, rej_out = model(input)
        predicted_labels = (cls_out > 0.5).float()

        loss_ae = criterion_ae(decoded, input)
        loss_cls = criterion_cls(cls_out, label)

        if rej_out < 0:
            prediction.append(2)
            combined_loss = loss_ae + loss_cls
        else:
            prediction.append(predicted_labels.item())
            loss_rej = rejection_loss(rej_out, loss_cls, 5, 1)
            combined_loss = loss_ae + loss_cls + loss_rej
        combined_loss.backward()
        optimizer.step()


print(prediction)
prediction_int = np.array(prediction, dtype=int)
print(np.sum(prediction_int == 2))
evaluate_metrics(index, torch.tensor(prediction))
