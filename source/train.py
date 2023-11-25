from loaddatasets import *
from model import *
from metric import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    path = '../data/generated_negtive_oversample_data.csv'
    data, labels, indexs = iterabel_dataset_generation(path)

    _, feature = data.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(feature).to(device)
    rejmodel = RejectionModel().to(device)

    criterion_ae = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    criterion_rej = nn.MSELoss()

    optimizer_clf = optim.Adam(model.parameters(), lr=0.001)
    optimizer_rej = optim.Adam(rejmodel.parameters(), lr=0.001)

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

    prediction_int = np.array(prediction, dtype=int)
    evaluate_metrics(labels, torch.tensor(prediction))
