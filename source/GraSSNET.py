import os
import math
import time

import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphAttentionLayer
import sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_curve,roc_auc_score
from scipy.special import lambertw
from typing import Optional
import warnings
warnings.simplefilter("ignore")

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = 'cuda'



data = pd.read_csv('../data/final.csv')
data = data.iloc[:50000]
selected_columns = [col for col in data.columns if col.startswith("Sensor")]
X = data[selected_columns].iloc[:, :-1]
# Use the ~ operator to select the columns that don't match the prefix
Y = data.loc[:, ~data.columns.isin(selected_columns)]
label_cols = Y.columns
X = X.values[:, np.newaxis, :]
Y = Y.values
X_train = X[:40000, :,:]
label_train = Y[:40000, :]
X_test = X[40000:, :,:]
label_test = Y[40000:, :]
class_numbers_train = [label_train[:,i].sum() for i in range(label_train.shape[1])]
meas_steps = [ms for ms in label_cols]
df_class_numbers_train = pd.DataFrame(np.asarray(class_numbers_train).reshape(1,label_train.shape[1]), columns = meas_steps)
class_numbers_test = [label_test[:,i].sum() for i in range(label_test.shape[1])]
meas_steps = [ms for ms in label_cols]
df_class_numbers_test = pd.DataFrame(np.asarray(class_numbers_test).reshape(1,label_test.shape[1]), columns = meas_steps)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, interval=1):
        self.data = data
        self.interval = interval
        self.labels = labels
        self.df_length = len(data)
        self.x_end_idx = self.get_x_end_idx()

    def __getitem__(self, index):
        idx = self.x_end_idx[index]
        train_data = self.data[idx]
        target_data = self.labels[idx]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(0, self.df_length)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'GCN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'GCN.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))

class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, hidden, device):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.multi = multi_layer
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.GRU = nn.GRU(self.time_step, self.unit, num_layers=1)

        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

        self.multihead_attn = nn.MultiheadAttention(self.time_step * self.multi, self.time_step)

        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(
            [self.unit, self.time_step * self.multi])  # drops the bias and gain., elementwise_affine=False
        self.fc1 = nn.Linear(self.time_step * self.unit * self.multi, self.time_step * self.unit * self.multi)
        self.fc2 = nn.Linear(self.time_step * self.unit * self.multi, hidden)

    def get_laplacian(self, graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x, indx):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.LeakyReLU(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    #     def spe_seq_cell(self, input):
    #         batch_size, k, input_channel, node_cnt, time_step = input.size()
    #         input = input.view(batch_size, -1, node_cnt, time_step)
    #         ffted = torch.rfft(input, 1, onesided=False)
    #         real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
    #         img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
    #         for i in range(3):
    #             real = self.GLUs[i * 2](real)
    #             img = self.GLUs[2 * i + 1](img)
    #         real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
    #         img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
    #         time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
    #         iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
    #         return iffted

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.view_as_real(torch.fft.fft(input, dim=1))
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.fft.irfft(torch.view_as_complex(time_step_as_inner), n=time_step_as_inner.shape[1], dim=1)
        return iffted

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs, requires_grad=True) * 0.1
        return inputs + noise

    def forward(self, x_emb):
        mul_L, _ = self.latent_correlation_layer(x_emb, 1)
        x = x_emb.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        gfted = torch.matmul(mul_L.unsqueeze(1), x.unsqueeze(1))
        gconv_input = self.spe_seq_cell(gfted)
        forecast_source = torch.sum(gconv_input, dim=1)
        forecast_source = self.layernorm(forecast_source)
        forecast_source = self.LeakyReLU(forecast_source)
        forecast_source = torch.flatten(forecast_source, 1)
        forecast_source = self.dropout(forecast_source)
        forecast_source = self.fc1(forecast_source)
        forecast = self.fc2(forecast_source)
        return forecast

class Model(nn.Module):
    def __init__(self, units, time_step, multi_layer, labels=1, dropout_rate=0.5, leaky_rate=0.2, vocab_size=629,
                 categorical=0, embedding_dim=16,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units - categorical
        self.time_step = time_step
        self.hidden = 64
        self.labels = labels
        self.multi_layer = multi_layer
        self.categorical = categorical

        self.stock_block = StockBlockLayer(self.time_step, self.unit, self.multi_layer, self.hidden, device)

        self.leakyrelu = nn.LeakyReLU(leaky_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_rate)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.outputnorm = nn.LayerNorm(self.hidden, elementwise_affine=False)

        self.label_output = nn.Sequential(
            nn.Linear(self.categorical * embedding_dim, self.hidden * 2),
            nn.Linear(self.hidden * 2, labels), )

        self.multihead_attn = nn.MultiheadAttention(1, 1)
        self.attnorm = nn.LayerNorm(self.labels)
        self.linear = nn.Linear(self.labels, self.labels)

        self.attentions = [GraphAttentionLayer(1, 8, dropout=dropout_rate / 5, alpha=0.2, concat=True) for _ in
                           range(multi_layer)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(8 * multi_layer, 1, dropout=dropout_rate / 5, alpha=0.2, concat=False)

        self.sensor = nn.Sequential(nn.Linear(self.hidden, labels), )

        self.to(device)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs, requires_grad=True) * 0.1
        return inputs + noise

    def forward(self, x, adj):
        B, _, _ = x.size()
        #         # ========================= Word Embedding ===================================
        #         x_cat_embedding = torch.flatten(self.embeddings(x[:, :1, :self.categorical].long()), start_dim=2)
        #         prediction_label = self.label_output(torch.flatten(x_cat_embedding, start_dim=1))

        #         # ========================= Sensor Feature Engineering =======================
        x_snsr = self.add_noise(x)  # (x[:, :, self.categorical:])
        x_emb1 = x_snsr
        prediction_sensor1 = self.leakyrelu(self.outputnorm(self.stock_block(x_emb1)))
        prediction = self.sensor(prediction_sensor1)

        # ========================= emerge branches ===================================
        attn, _ = self.multihead_attn(prediction.permute(1, 0).unsqueeze(2), prediction.permute(1, 0).unsqueeze(2),
                                      prediction.permute(1, 0).unsqueeze(2))
        prediction = self.leakyrelu(self.linear(self.attnorm(prediction + self.dropout(attn.squeeze().permute(1, 0)))))

        # ========================= Label Correlation =========================
        predictions = torch.cat([att(prediction.unsqueeze(2), adj) for att in self.attentions], dim=2)
        predictions = self.out_att(predictions, adj).squeeze()

        # ========================= Softmax Constraints =========================
        predictions = torch.split(predictions, 2, dim=1)
        predictions = [self.softmax(pred) for pred in predictions]
        result = torch.cat(predictions, 1)

        return result

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, label_smoothing=0.0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True, Asymmetric=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.Asymmetric = Asymmetric

    def _process_labels(self, labels, label_smoothing=None, dtype=None):
        labels = labels.type(dtype)
        if label_smoothing is not None:
            labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
        return labels

    def forward(self, x, y, pos_weight=None, neg_weight=None, mask=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        y_smooth = self._process_labels(y, label_smoothing=self.label_smoothing, dtype=x.dtype)

        # Calculating Probabilities
        xs_pos = x
        xs_neg = 1 - x

        # Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y_smooth * (xs_neg ** self.gamma_pos) * torch.log(xs_pos.clamp(min=self.eps)) * pos_weight
        los_neg = (1 - y_smooth) * (xs_pos ** self.gamma_neg) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        if self.Asymmetric:
            # Asymmetric Focusing
            if self.gamma_neg > 0 or self.gamma_pos > 0:
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(False)
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(True)
                loss *= one_sided_w

        return -loss


def inference(model, dataloader, adj, device):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            forecast_result = model(inputs, adj)
            forecast_set.append(forecast_result)
            target_set.append(target)
    return torch.cat(forecast_set, 0), torch.cat(target_set, 0)


def EVAL(label_pred, label_true):
    mask = torch.ones(label_true.shape[0], label_true.shape[1]).to(device)
    index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).to(device)
    mask = mask.index_copy_(1, index, label_true[:, ::2] + label_true[:, 1::2])
    mask = mask.index_copy_(1, index + 1, label_true[:, ::2] + label_true[:, 1::2])
    loss_valid = forecast_loss_valid(label_pred, label_true) * mask

    label_pred_prob = label_pred
    label_pred_prob = label_pred_prob.masked_fill(label_true == 0, 0)
    acc = sum(sum(label_pred_prob > 0.5)[::2]).item() / sum(sum(label_true > 0)[::2]).item()
    recall = sum(sum(label_pred_prob > 0.5)[1::2]).item() / sum(sum(label_true > 0)[1::2]).item()

    return acc, recall, loss_valid.sum() / mask.sum()


def get_correlation(label_train):
    label_train_adj = label_train
    arr = np.zeros((label_train_adj.shape[1], label_train_adj.shape[1]))
    for i in range(label_train_adj.shape[1]):
        for j in range(label_train_adj.shape[1]):
            pairs = label_train_adj[:, i] + label_train_adj[:, j]
            pairs_count = np.count_nonzero(pairs == 2)
            arr[i, j] = pairs_count

    class_numbers = np.asarray([label_train_adj[:, i].sum() for i in range(label_train_adj.shape[1])])
    zero_classes = np.squeeze(np.asarray(np.where(class_numbers == 0)))
    if len(zero_classes):
        print('zero_classes:', zero_classes)
    class_numbers[class_numbers == 0] = 1
    correlation = arr / (class_numbers[:, np.newaxis])

    correlation = correlation - np.eye(label_train_adj.shape[1])

    correlation[correlation < 0.2] = 0
    correlation[correlation >= 0.2] = 1
    correlation = correlation + np.identity(label_train_adj.shape[1], np.int)

    return correlation


correlation = get_correlation(label_train)
meas_steps = [ms for ms in label_cols]

train_data_set = Dataset(X_train, label_train)
test_data_set = Dataset(X_test, label_test)
valid_data_set = Dataset(X_test, label_test)
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=1024, drop_last=False, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=256, drop_last=False, shuffle=False, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_data_set, batch_size=256, drop_last=False, shuffle=False, num_workers=0)
print('[Train] # batches->', len(train_loader), '\n[Test] # batches->', len(test_loader))

criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=2, label_smoothing=0.0, clip=0.05, disable_torch_grad_focal_loss=True,
                           Asymmetric=False)
forecast_loss_valid = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

# hyper parameters
multi = 4
units = X_train.shape[-1]
time_step = X_train.shape[-2]
labels_num = label_train.shape[1]
# categorical_num = 6
# embedding_dim = 16
print('# nodes:',units, '# time_step:', time_step, '# labels:', labels_num)

model = Model(units=units, time_step=time_step, multi_layer=multi, labels=labels_num, device=device)
model.to(device)

positives = torch.tensor(df_class_numbers_train.values.astype(np.float32))
negtives = (label_train.shape[0] - positives)
pos_weight = negtives / positives
pos_weight = pos_weight.squeeze()
print('pos_weight: ', pos_weight)

total_params = 0
for name, parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    param = parameter.numel()
    total_params += param
print(f"Total Trainable Params: {total_params}")

my_optim = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
# my_optim = torch.optim.RMSprop(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(my_optim, 'min', patience=10, factor=0.8)

train_loss = []
valid_loss = []
acc_test = []
recall_test = []
best_performance = 0.0
min_loss_train = float('inf')
min_loss_valid = float('inf')
clipping_value = 1
loss_no_improve_train = 0
loss_no_improve_valid = 0
epochs_no_improve = 0
early_stop = 30
validate_freq = 1
iteration = 0

adj = torch.from_numpy(correlation).to(device)

for epoch in range(1000):
    epoch_start_time = time.time()
    model.train()
    loss_total = 0
    cnt = 0
    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        mask = torch.ones(target.shape[0], target.shape[1]).to(device)
        index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).to(device)
        mask = mask.index_copy_(1, index, target[:, ::2] + target[:, 1::2])
        mask = mask.index_copy_(1, index + 1, target[:, ::2] + target[:, 1::2])

        model.zero_grad()
        forecast = model(inputs, adj)

        # todo: binary_focal_loss with label smoothing:
        #       binary_focal_loss: Setting γ > 0 reduces the relative loss for well-classified examples (pt > .5), putting more focus on hard, misclassified examples
        #                          γ = 0: standard cross entropy
        #       label smoothing: a regularization technique for classification problems to prevent the model from predicting the training examples too confidently.
        loss = criterion(forecast, target, pos_weight.to(device)) * mask
        loss = loss.sum() / (mask.sum())
        """
        # todo: BCEWITHLOGITSLOSS: This loss combines a Sigmoid layer and the BCELoss in one single class. Each labels are independent
        forecast_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none').to(device)  # pos_weight=pos_weight, weight=sample_weights, 
        loss = forecast_loss(forecast, target)*mask
        loss = loss.sum() / mask.sum()
        """
        forecast_loss_un = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        mask_unlabel = forecast.ge(0.95).float()
        loss_unlabel = (forecast_loss_un(forecast, torch.round(forecast)) * mask_unlabel).mean()

        loss = loss + loss_unlabel

        cnt += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        my_optim.step()
        loss_total += float(loss)

    if (epoch + 1) % validate_freq == 0:

        train_loss.append(loss_total / cnt)
        is_best_for_now_minloss = False
        is_best_for_now_maxperformance = False

        label_pred_test, label_true_test = inference(model, valid_loader, adj, device)
        acc_t, recall_t, loss_valid_t = EVAL(label_pred_test, label_true_test)
        acc_test.append(acc_t)
        recall_test.append(recall_t)
        valid_loss.append(loss_valid_t)

        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | acc_valid {:5.4f}, | recall_valid {:5.4f} | best_validate_acc {:5.4f} | min_loss_validate {:5.4f}'.  # | lr {:5.4f}
            format(epoch + 1, (time.time() - epoch_start_time), loss_total / cnt, acc_t, recall_t, best_performance,
                   min_loss_valid))

        # ================================================================
        # learning rate scheduale by validation loss
        my_lr_scheduler.step(loss_valid_t)

        # ================================================================
        if (acc_t + recall_t) > best_performance and epoch >= 10:
            epochs_no_improve = 0
            best_performance = (acc_t + recall_t)
            is_best_for_now_maxperformance = True
        else:
            epochs_no_improve += 1

            # ===================================================================
        if loss_valid_t < min_loss_valid and epoch >= 10:
            loss_no_improve_valid = 0
            min_loss_valid = loss_valid_t
            is_best_for_now_minloss = True
        else:
            loss_no_improve_valid += 1

        # save the best model by minimum validation loss / highest performance of validation
        if is_best_for_now_minloss:
            result_file = os.path.join('output', 'stem_simple_best_minloss')
            save_model(model, result_file)

        if is_best_for_now_maxperformance:
            result_file = os.path.join('output', 'stem_simple_best_performance')
            save_model(model, result_file)

        if epochs_no_improve == early_stop and epoch >= 10:  # loss_no_improve_valid
            print('Early stopping!')
            break
        else:
            continue

result_file = os.path.join('output', 'stem_simple_random')
save_model(model, result_file)