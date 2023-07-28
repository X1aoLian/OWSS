import numpy as np
from matplotlib import pyplot as plt, pylab
from scipy.spatial import distance
import itertools


def list_of_groups(list_info, per_list_len):
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

def PairDotsDistance(Dots, matrix, start, end):
    numOfDots = end - start
    if end == 2096:
        for i in range(0, numOfDots - 1):
            for j in range(i + 1, numOfDots):
                matrix[i][j] = np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2))
    else:
        for j in range(1000, numOfDots):
            for i in range(0, j):
                matrix[i][j] = np.sqrt(np.sum((Dots[:, i] - Dots[:, j]) ** 2))

    return  matrix

def ori_PairDotsDistance(Dots, start, end, Distance):
    DistanceMat = []
    numOfDots = end - start

    window_size = 1000
    if start == 0:
        for i in range(start, numOfDots - 1):
            for j in range(i+1, numOfDots):
                DistanceMat.append(i)
                DistanceMat.append(j)
                DistanceMat.append(np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2)))
        return np.array(list_of_groups(DistanceMat, 3))
    else:
        for i in range(0, end - 1):
            if i < window_size + start:
                for j in range(end - start, end):
                    DistanceMat.append(i)
                    DistanceMat.append(j)
                    DistanceMat.append(np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2)))
            else:
                for j in range(i + 1, end):
                    DistanceMat.append(i)
                    DistanceMat.append(j)
                    DistanceMat.append(np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2)))


        return np.concatenate((Distance, np.array(list_of_groups(DistanceMat, 3))), axis = 0)

    #print(DistanceMat)


def DensityPeaks(xi, precent, start, end, matrix):

    xiT = xi.T
    dist = PairDotsDistance(xiT, matrix, start, end)
    # print('average percentage of neighbours (hard coded): %5.6f\n', precent)

    N = (end * end)/2 - end
    position = round(N * precent / 100, 0)
    x = np.delete(dist.flatten(), np.where(dist.flatten() == 0))
    dc = np.sort(x)[int(position)]

    rho = np.zeros((1, int(end))) # [i for i in range(1, ND)]

    for i in range(0, int(end) - 1):
        for j in range(i + 1, int(end)):
            rho[0, i] = rho[0, i] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
            rho[0, j] = rho[0, j] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
    maxd = max(np.max(dist, axis=0))


    ordrho = np.argsort(-rho,axis=1)
    delta = np.zeros((int(end)))
    delta[ordrho[0,0] - 1] = -1.
    delta = delta.reshape(-1, delta.shape[0])
    nneigh = np.zeros((int(end)))
    nneigh[ordrho[0,0] - 1] = 0
    nneigh = nneigh.reshape(-1, nneigh.shape[0])

    for ii in range(1, int(end)):
        delta[0, ordrho[0, ii]] = maxd
        for jj in range(0, ii - 1):
            if (dist[ordrho[0, ii], ordrho[0, jj]] < delta[0, ordrho[0, ii]]):
                delta[0, ordrho[0,ii]] = dist[ordrho[0, ii], ordrho[0, jj]]
                nneigh[0, ordrho[0,ii]] = ordrho[0, jj]

    delta[0, ordrho[0, 0]] = maxd
    rho = rho.T
    nneigh = nneigh.T
    delta = delta.T
    b = np.argmin(nneigh)
    nneigh[b,0] = b
    return rho, delta, nneigh, dist


def depthmin(x,n,depth):
    if depth > 2:
        return depth
    else:
        if n[x] != 0:
            return depth
        else:
            depth += 1
            return depthmin(int(n[x]),n, depth)
def centerindex(delta):
    center1 = np.argsort(delta, axis=0)[-1]
    center2 = np.argsort(delta, axis=0)[-2]
    center3 = np.argsort(delta, axis=0)[-3]
    center4 = np.argsort(delta, axis=0)[-4]
    center5 = np.argsort(delta, axis=0)[-5]
    center6 = np.argsort(delta, axis=0)[-6]
    center7 = np.argsort(delta, axis=0)[-7]
    center8 = np.argsort(delta, axis=0)[-8]
    center9 = np.argsort(delta, axis=0)[-9]
    center10 = np.argsort(delta, axis=0)[-10]
    center = [ center1,center2,center3,center4,center5]
    return center

def obtainlabel(x,n,label):
    if label[n[x]] == 0:
        return obtainlabel(int(n[x]),n,label)
    else:
        return label[n[x]]

def clustering(dw, gamma=0.02, decision_graph=False,
               alpha=0.5, beta=0.5, cluster=False, metric='euclidean'):
    # step1: pairwise distance
    condensed_distance = distance.pdist(dw, metric=metric)
    d_c = np.sort(condensed_distance)[int(len(condensed_distance) * gamma)]
    redundant_distance = distance.squareform(condensed_distance)
    # step2: calculate local density
    rho = np.sum(np.exp(-(redundant_distance / d_c) ** 2), axis=1)

    # step3: calculate delta
    order_distance = np.argsort(redundant_distance, axis=1)
    delta = np.zeros_like(rho)
    nn = np.zeros_like(rho).astype(int)
    for i in range(len(delta)):
        mask = rho[order_distance[i]] > rho[i]
        if mask.sum() > 0:  # not the highest density point
            nn[i] = order_distance[i][mask][0]
            delta[i] = redundant_distance[i, nn[i]]
        else:  # the highest density point
            nn[i] = order_distance[i, -1]
            delta[i] = redundant_distance[i, nn[i]]
    if decision_graph:
        plt.scatter(rho, delta)
        plt.show()

    rho_c = min(rho) + (max(rho) - min(rho)) * alpha
    delta_c = min(delta) + (max(delta) - min(delta)) * beta
    centers = np.where(np.logical_and(rho > rho_c, delta > delta_c))[0]

    if not cluster:
        return centers
    else:
        labels = np.zeros_like(rho)
        for i, v in enumerate(centers):
            labels[v] = i
        order_rho = np.argsort(rho)[::-1]
        for p in order_rho:
            if p not in centers:
                labels[p] = labels[nn[p]]
        return centers, labels

def ori_DensityPeaks(xi, precent, start, end, DistanceMat):

    xiT = xi.T
    xx = ori_PairDotsDistance(xiT, start, end, DistanceMat)
    xxT = xx

    ND = np.max(xxT[:, 1]) + 1
    NL = np.max(xxT[:, 0]) + 1

    if NL > ND:
        ND = NL
    N = xxT.shape[0]

    dist = np.zeros((int(ND),int(ND)))

    for i in range(0, N):
        ii = xxT[i, 0]
        jj = xxT[i, 1]

        dist[int(ii), int(jj)] = xxT[i, 2]
        dist[int(jj), int(ii)] = xxT[i, 2]

    # print('average percentage of neighbours (hard coded): %5.6f\n', precent)

    position = round(N * precent / 100, 0)

    sda = np.sort(xxT[:, 2])

    dc = sda[int(position)]

    rho = np.zeros((1, int(ND))) # [i for i in range(1, ND)]

    for i in range(0, int(ND) - 1):
        for j in range(i + 1, int(ND)):
            rho[0, i] = rho[0, i] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
            rho[0, j] = rho[0, j] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))

    maxd = max(np.max(dist, axis=0))

    rho_sorted = np.sort(rho, axis=1)

    rho_list = list(rho_sorted)
    rho_list = list(itertools.chain.from_iterable(rho_list))
    rho_list.reverse()
    rho_list = np.array(rho_list)
    rho_list = rho_list.reshape(-1, rho_list.shape[0])
    rho_sorted = rho_list
    ordrho = np.argsort(-rho,axis=1)

    delta = np.zeros((int(ND)))
    delta[ordrho[0,0] - 1] = -1.
    delta = delta.reshape(-1, delta.shape[0])

    nneigh = np.zeros((int(ND)))
    nneigh[ordrho[0,0] - 1] = 0
    nneigh = nneigh.reshape(-1, nneigh.shape[0])

    for ii in range(1, int(ND)):
        delta[0, ordrho[0, ii]] = maxd
        for jj in range(0, ii - 1):
            if (dist[ordrho[0, ii], ordrho[0, jj]] < delta[0, ordrho[0, ii]]):
                delta[0, ordrho[0,ii]] = dist[ordrho[0, ii], ordrho[0, jj]]
                nneigh[0, ordrho[0,ii]] = ordrho[0, jj]

    delta[0, ordrho[0, 0]] = maxd

    rho_sorted = rho_sorted.T
    rho = rho.T
    ordrho = ordrho.T
    nneigh = nneigh.T
    delta = delta.T

    b = np.argmin(nneigh)
    nneigh[b,0] = b

    return rho, delta, nneigh, xx


