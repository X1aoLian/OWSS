from torch import nn
from torch.autograd import Variable
from metric import *
import torch.nn
import torch.nn.functional as F


class RejectModel(nn.Module):
    def __init__(self,original_size):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""
        super(RejectModel, self).__init__()
        self.original_size = original_size

        self.Linear1 = nn.Linear(self.original_size, 32)
        self.Linear2 = nn.Linear(32,1)
    def forward(self, x):
        #encoder = self.encoder(x)
        x = self.Linear1(x)
        x = F.relu(x)
        rejection = self.Linear2(x)
        return  rejection


class MLP(nn.Module):
    def __init__(self,original_size):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""
        super(MLP, self).__init__()
        self.original_size = original_size
        self.Linear1 = nn.Linear(self.original_size, 64)
        self.Linear2 = nn.Linear(64, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #encoder = self.encoder(x)
        x = self.Linear1(x)
        x = self.relu(x)
        out = self.Linear2(x)
        return  out

    def distanceloss(self, centers, pairdistance,lable):
        loss = 0
        closetcenter = []
        index = np.where(lable != 0)[0]

        for i in (index):
            for center in centers:
                if lable[i] == lable[center] and int(i) < int(center):
                    position = np.where((pairdistance[:, 0] == i) & (pairdistance[:, 1] == center))
                    position = position[0]
                    closetcenter.append(pairdistance[position[0], 2])
                elif lable[i] == lable[center] and int(i) > int(center):
                    position = np.where((pairdistance[:, 1] == i) & (pairdistance[:, 0] == center))
                    position = position[0]
                    closetcenter.append(pairdistance[position[0], 2])
                else:
                    closetcenter.append(1)

            loss = loss + min(closetcenter)#why!!!!!!!!!! two boolean values
            closetcenter = []

        loss = Variable(torch.Tensor(np.array([loss])), requires_grad=True)
        return loss

    def densitypeaks(self, data, start, end, DistanceMat):

        rho, delta, neighbor, DistanceMat = ori_DensityPeaks(data,5, start, end, DistanceMat)
        centers = centerindex(delta)
        return  rho, delta, neighbor, DistanceMat, centers

    def update(self, net, optimizer, centers, DistanceMat, mask_label):
        loss = net.distanceloss(centers, DistanceMat, mask_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
