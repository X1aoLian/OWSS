
from torch.autograd import Variable
from datasets import *
from model import  RejectModel, MLP



import time


def warm(data, mask_label, start_windows, end_window, DistanceMat, latent_size, weightrejection, prediction, training_epoch, path, number):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, feature_num = data.size()
    net = MLP(feature_num).to(device)

    out = net(data[start_windows:end_window])
    rho, delta, neighbor, DistanceMat, centers = net.densitypeaks(out.cpu().detach().numpy(), start_windows,
                                                                   end_window, DistanceMat)

    optimizer1 = torch.optim.Adam(net.parameters(), lr=0.01)

    for center in centers:
        if label[center]  == 0:
            label[center] = mask_label[neighbor[center]]
    AE = RejectModel(latent_size).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.001)
    for _ in range(1):
        z = out.detach().clone()
        for _ in range(1):
            rejectionlevel = AE(z)
            buffer = np.array(np.where(rejectionlevel.cpu().detach().numpy().flatten() > 0))
            for i in range(len(prediction)):
                if np.isin(i, buffer):
                    if mask_label[neighbor[i]] != 0:
                        prediction[i] = mask_label[neighbor[i]]
                    else:
                        prediction[i] = mask_label[neighbor[int(neighbor[i])]]

            firstterm = 1 + 0.5 * (torch.sum(rejectionlevel) / (end_window - start_windows) - torch.sum(torch.mul(prediction, mask_label)))
            secondterm = weightrejection * (1 - (1 / (2 * weightrejection - 1) * torch.sum(rejectionlevel) / (end_window - start_windows)))
            if firstterm > 0 or secondterm > 0:
                if firstterm > secondterm:
                    rejectionloss = firstterm
                else:
                    rejectionloss = secondterm
            else:
                rejectionloss = Variable(torch.Tensor([0.1]), requires_grad=True)

            loss = rejectionloss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  #
            optimizer.step()

        net.update(net, optimizer1, centers, DistanceMat, prediction.cpu().numpy().copy())

    np.save('../results/'+path+'/OOFLS/' + 'prediction' + str(number) +str(0)+'.npy', prediction.cpu().numpy().copy())

    return net, AE, DistanceMat, prediction


def train(data, label,model_layers, training_epoch, latent_size, label_rate, path, number):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    start_windows = 0
    end_window = warmup[path]
    weightrejection = 10
    mask_label = label.clone().to(device)
    DistanceMat = []
    prediction = torch.zeros(len(label)).to(device)

    net, AE, matrix, prediction = warm(data,mask_label,start_windows,end_window,DistanceMat,latent_size, weightrejection,prediction, training_epoch, path, number)

    for epoch in range(20):
        start_windows = gap[path]*(epoch+1) +0
        end_window = gap[path]*(epoch+1) +warmup[path]
        out = net(data[:end_window])
        rho, delta, neighbor, matrix, centers = net.densitypeaks(out.cpu().detach().numpy(), start_windows,
                                                                       end_window, matrix)

        optimizer1 = torch.optim.SGD(net.parameters(), lr=0.001)


        optimizer = torch.optim.SGD(AE.parameters(), lr=0.001)
        for _ in range(1):
            z = out.detach().clone()
            for _ in range(1):
                rejectionlevel = AE(z[:end_window])
                buffer = np.array(np.where(rejectionlevel.cpu().detach().numpy().flatten() > 0))
                for i in range(len(prediction[:end_window])):
                    if np.isin(i, buffer):
                        if mask_label[:end_window][neighbor[i]] != 0:
                            prediction[i] = mask_label[:end_window][neighbor[i]]
                        else:
                            prediction[i] = mask_label[:end_window][neighbor[int(neighbor[i])]]

                firstterm = 1 + 0.5 * (torch.sum(rejectionlevel)  - torch.sum(torch.mul(prediction[:end_window], mask_label[:end_window])))
                secondterm = weightrejection * (1 - (1 / (2 * weightrejection - 1 ) * torch.sum(rejectionlevel)))
                if firstterm > 0 or secondterm > 0:
                    if firstterm > secondterm:
                        rejectionloss = firstterm
                    else:
                        rejectionloss = secondterm
                else:
                    rejectionloss = Variable(torch.Tensor([0.1]), requires_grad=True)

                loss = rejectionloss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  #
                optimizer.step()
            net.update(net, optimizer1, centers, matrix, prediction[:end_window].cpu().numpy().copy())

        np.save('../results/'+path+'/OOFLS/' + 'prediction' + str(number) +str(epoch+1)+'.npy', prediction.cpu().numpy().copy())

    return prediction







if __name__ == '__main__':
    label_rate = 0.2
    model_layers = 4
    training_epoch = 5
    reduced_size =  20
    random_seed = 22
    warmup = {'musk': 196, 'mnist': 175, 'optdigits': 219, 'satimage': 194, 'reuter': 3100}
    gap = {'musk': 100, 'mnist': 200, 'optdigits': 185, 'satimage': 170, 'reuter': 100}

    for path in [ 'musk', ]:
        if path == 'musk':
            data, label, x = musk_normalization(random_seed, label_rate)
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        mask_label = torch.Tensor(x)

        for i in range(5):
            prediction = train(data, mask_label, model_layers, training_epoch, reduced_size, label_rate, path=path, number=11)





