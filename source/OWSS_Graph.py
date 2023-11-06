import os
import math
import time

import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score
from scipy.special import lambertw
from typing import Optional
import warnings
warnings.simplefilter("ignore")

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = 'cuda'
else:
    device = 'cpu'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # print(h.shape, self.W.shape)  # torch.Size([1024, 22, 1]) torch.Size([1, 8])
        Wh = h.matmul(self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features) # Wh torch.Size([1024, 22, 8])
        a_input = self._prepare_attentional_mechanism_input(Wh)  # a_input torch.Size([1024, 22, 22, 16])
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # e torch.Size([1024, 22, 22])
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # attention: torch.Size([1024, 22, 22])

        attention = F.softmax(attention, dim=2)  #
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

        """
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        """

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (batch, N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (batch, N * N, 2 * out_features)

        return all_combinations_matrix.view(Wh.size()[0], N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


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
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

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
        prediction = self.classifier(prediction_sensor1)

        return prediction

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
    #if len(zero_classes):
    #    print('zero_classes:', zero_classes)
    class_numbers[class_numbers == 0] = 1
    correlation = arr / (class_numbers[:, np.newaxis])

    correlation = correlation - np.eye(label_train_adj.shape[1])

    correlation[correlation < 0.2] = 0
    correlation[correlation >= 0.2] = 1
    correlation = correlation + np.identity(label_train_adj.shape[1], np.int64)

    return correlation





data = pd.read_csv('../data/final.csv')
data = data.iloc[:50000]
selected_columns = [col for col in data.columns if col.startswith("Sensor")]
X = data[selected_columns].iloc[:, :-1]
# Use the ~ operator to select the columns that don't match the prefix
Y = data.loc[:, ~data.columns.isin(selected_columns)]


df = pd.read_csv('../data/generated_negtive_oversample_data.csv')
X = df.filter(regex='^Sensor')
df['mapped_label'] = df['new_label'].apply(lambda x: x if x in [5, 6] else 7)
# 生成独热编码
Y= pd.get_dummies(df['mapped_label']).astype(int)

# 重命名列以反映新的标签
Y.columns = [f'Label{i}-fail' for i in Y.columns]

# 为每个fail列添加一个pass列，所有值初始化为0
for i in range(3):
    Y.insert(2*i, f'Label{i}-pass', 0)


label_cols = Y.columns
X = X.values[:, np.newaxis, :]
Y = Y.values
X_train = X[:40000, :,:]

label_train = Y[:40000, :]

class_numbers_train = [label_train[:,i].sum() for i in range(label_train.shape[1])]
meas_steps = [ms for ms in label_cols]
df_class_numbers_train = pd.DataFrame(np.asarray(class_numbers_train).reshape(1,label_train.shape[1]), columns = meas_steps)
meas_steps = [ms for ms in label_cols]
correlation = get_correlation(label_train)
meas_steps = [ms for ms in label_cols]


train_data_set = Dataset(X_train, label_train)

train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=2, drop_last=False, shuffle=True, num_workers=0)




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
positives= torch.where(positives == 0, torch.tensor(1.0), positives)

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

prediction_list = []
target_list = []
adj = torch.from_numpy(correlation).to(device)

for epoch in range(1):
    epoch_start_time = time.time()
    model.train()
    loss_total = 0
    cnt = 0
    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.to(device)
        target = target.to(device)
        target_list.append(target[0].detach().cpu().numpy())
        target_list.append(target[1].detach().cpu().numpy())

        index = torch.tensor([0, 2, 4,]).to(device)

        forecast = model(inputs, adj)
        print(forecast)
        #prediction_list.append(forecast[0].detach().cpu().numpy())
        #prediction_list.append(forecast[1].detach().cpu().numpy())



#result_file = os.path.join('output', 'stem_simple_random')
#save_model(model, result_file)
def reassign_labels(samples):
    new_labels = []
    for sample in samples:
        # 取出所有奇数位置的值，即所有pos的值
        pos_values = sample[1::2]
        # 找到最大pos值的索引
        max_pos_index = pos_values.argmax()
        # 该索引对应的是第几对
        label = max_pos_index
        new_labels.append(label)
    return new_labels

def evaluate_metrics(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # 1. Calculate accuracy for predicted values of 0 or 1
    mask = predicted_labels != 2
    accuracy = (true_labels[mask] == predicted_labels[mask]).mean()

    # 2.
    true_binary = (true_labels == 2)
    predicted_binary = (predicted_labels == 2)

    f1 = f1_score(true_binary, predicted_binary)
    recall = recall_score(true_binary, predicted_binary)
    precision = precision_score(true_binary, predicted_binary)

    print(f"Accuracy (only for predicted 0 or 1): {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")


new_label = reassign_labels(target_list)
new_prediction = reassign_labels(prediction_list)
evaluate_metrics(new_label, new_prediction)

