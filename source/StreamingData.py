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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def transform_label(value):
    if value == 6:
        return 0
    elif value == 5 :
        return 1
    else:
        return 2

def iterabel_dataset_generation(path):
    df = pd.read_csv(path)
    df['new_label'] = df['new_label'].apply(transform_label)
    sensor_data = df.filter(regex='^Sensor')
    labels = df['new_label']

    # 使用pd.get_dummies将new_label列转换为独热编码形式
    label_dummies = pd.get_dummies(labels).astype(int)
    label_dummies.columns = [f'Label{i}-fail' for i in range(label_dummies.shape[1])]
    # Normalize the data
    index = df['new_label'].values

    data = df.iloc[:, :-1]
    num_rows, num_cols = data.shape
    rows_per_chunk = num_rows // 10
    cols_per_chunk = num_cols // 10

    for i in range(9):
        start_row = i * rows_per_chunk
        end_row = start_row + rows_per_chunk
        end_col = (i + 1) * cols_per_chunk

        # 将超出范围的列设置为 0
        data.iloc[start_row:end_row, end_col:] = 0

    # 处理最后一个块（它可能有多于 rows_per_chunk 的行）
    data.iloc[9 * rows_per_chunk:, :] = data.iloc[9 * rows_per_chunk:, :]
    #dataset = Dataset.from_pandas(data)
    label = df.iloc[:, -1].values.astype(int)

    data, label, index = shuffle(data, label, index, random_state=1020)
    #iterable_dataset = dataset.to_iterable_dataset()
    np.save('true_label',label)
    np.save('true_index',label)
    return data.values, label, index

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(94, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 94),
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
        return encoded, decoded, cls_out

class RejectionModel(nn.Module):
    def __init__(self):
        super(RejectionModel, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def rejection_loss(rejection_value, loss, D ,lambda_):

    first_term = 0.5 * rejection_value + loss - D
    secondterm = lambda_ * (1 - (1 / (2 * lambda_ - 1) * rejection_value))

    result = torch.max(torch.max(first_term, secondterm), torch.tensor(0).to(first_term.device))
    result = torch.tensor([result], requires_grad=True).to(device)
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
rejmodel = RejectionModel().to(device)

criterion_ae = nn.MSELoss()
criterion_cls = nn.BCELoss()
criterion_rej = nn.MSELoss()

optimizer_clf = optim.Adam(model.parameters(), lr=0.001)
optimizer_rej = optim.Adam(rejmodel.parameters(), lr=0.001)

path = '../data/generated_negtive_oversample_data.csv'
data, labels, indexs = iterabel_dataset_generation(path)
indexs = (indexs == 2)
data, labels, indexs = torch.tensor(data), torch.tensor(labels), torch.tensor(indexs)
prediction = []



for i, x in enumerate(data):
    input = x.to(device).float()
    label = labels[i].to(device).float().unsqueeze(0)
    index = indexs[i].to(device).float().unsqueeze(0)
    encoded, decoded, cls_out = model(input)
    rej_out = rejmodel(encoded)
    predicted_label = (cls_out > 0.5).float()
    predicted_rejection = (rej_out > 0.5).float()
    loss_ae = criterion_ae(decoded, input)


    if predicted_rejection == 0:
        optimizer_rej.zero_grad()
        loss_rej = criterion_rej(rej_out, index)
        loss_rej.backward(retain_graph=True)
        optimizer_rej.step()
        optimizer_clf.zero_grad()
        if label != 2:
            loss_cls = criterion_cls(cls_out, label)
        else:
            inverted_labels = 1 - predicted_label
            loss_cls = criterion_cls(cls_out, inverted_labels)
        combined_loss = loss_ae + loss_cls
        combined_loss.backward()
        optimizer_clf.step()
        prediction.append(predicted_label.item())

    else:
        optimizer_rej.zero_grad()
        loss_rej = criterion_rej(rej_out, index)
        loss_rej.backward(retain_graph=True)
        optimizer_rej.step()
        prediction.append(2)


np.save('prediction_owss_1.npy',np.array(prediction, dtype=int))




prediction_int = np.array(prediction, dtype=int)
evaluate_metrics(labels, torch.tensor(prediction))
print(sum(labels == 0))
print(sum(labels == 1))
print(sum(labels == 2))
print(sum(prediction_int == 2))