import numpy as np
from knngraph import *

def getAffinityMaxtrix(Vc,Kc,W):
    nc = len(Vc)

    affinity = np.zeros([nc,nc])

    for i in range(nc):
        for j in range(i+1,nc):
            ij = np.ix_(Vc[i],Vc[j])
            ji = np.ix_(Vc[j],Vc[i])

            W_ij, W_ji = W[ij], W[ji]
            Ci, Cj = len(Vc[i]),len(Vc[j])

            ones_i = np.ones((Ci,1))
            affinity[i][j] = np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i)
            affinity[j][i] = affinity[i][j]
    return affinity

def getAffinityBtwCluster(C1, C2, Kc, W):


    ij = np.ix_(C1, C2)
    ji = np.ix_(C2, C1)

    W_ij, W_ji = W[ij], W[ji]
    Ci, Cj = len(C1), len(C2)

    ones_i = np.ones((Ci, 1))
    affinity = np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i)
    #print(affinity)
    return affinity[0,0]

def getNeighbor(Vc, Kc, W):
    Ns, As = [], []
    #time1 = time.time()
    #A = getAffinityMaxtrix(Vc, Kc, W)
    #np.save('A_MNIST.npy',A)
    A = np.load('A_MNIST.npy')
    #print("Affinity time : ", time.time() - time1)
    for i in range(len(A)):
        As.append([x for x in sorted(list(A[i]))[-1 * Kc:] if x > 0])
        n = len(As[i])
        if n==0:
            Ns.append([])
        else:
            Ns.append(A[i].argsort()[-1*n:].tolist())

    return Ns,As


def AGDL(data, targetClusterNum, Ks, Kc):
    cluster = k0graph(data)
    print("k0graph complete")

    #W = w_matrix(data, Ks)
    W = np.load("W_" + str(Ks) + "_MNIST.npy")
    print("neighbor")
    neighborSet, affinitySet = getNeighbor(cluster, Kc, W)
    currentClusterNum = len(cluster)

    while currentClusterNum > targetClusterNum:

        max_affinity = 0
        for i in range(len(neighborSet)):
            if len(affinitySet[i])==0:
                continue
            aff = max(affinitySet[i])
            if aff > max_affinity:
                j = int(neighborSet[i][affinitySet[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            print("index alias")


        print(len(cluster[max_index2]) == 0, max_affinity)
        cluster[max_index1].extend(cluster[max_index2])
        cluster[max_index2] = []


        neighborSet[max_index2] = []
        affinitySet[max_index2] = []



        for i in range(len(neighborSet)):
            if max_index1 in neighborSet[i]:
                index = neighborSet[i].index(max_index1)
                affinitySet[i][index] = getAffinityBtwCluster(cluster[i], cluster[max_index1], Kc, W)

            if max_index2 in neighborSet[i] and max_index1 != i:
                index = neighborSet[i].index(max_index2)
                del neighborSet[i][index]
                del affinitySet[i][index]
                if max_index1 not in neighborSet[i] and max_index1 != i:
                    neighborSet[i].append(max_index1)
                    affinitySet[i].append(getAffinityBtwCluster(cluster[i], cluster[max_index1], Kc, W))

        neighborSet[max_index1].extend(neighborSet[max_index2])
        neighborSet[max_index1] = list(set(neighborSet[max_index1]))
        affinitySet[max_index1] = []

        if currentClusterNum < 50:
            print(neighborSet[max_index1])
        # Fine the Kc-nearest clusters for Cab

        for i in range(len(neighborSet[max_index1])):

            index = neighborSet[max_index1][i]
            affinitySet[max_index1].append(getAffinityBtwCluster(cluster[index], cluster[max_index1], Kc, W))

        currentClusterNum = currentClusterNum - 1


    reduced_cluster = []
    for i in range(len(cluster)):
        if len(cluster[i]) != 0:
            reduced_cluster.append(cluster[i])
    #print("Cluster : ", Cluster)
    length = 0
    for i in range(len(reduced_cluster)):
        length += len(reduced_cluster[i])
    print("Data number, Cluster number : ", length, len(reduced_cluster))

    return reduced_cluster



if __name__ == '__main__':
    sample = np.array([[1, 1],[1, 2], [2, 1], [2, 2], [3, 4], [1, 7], [3,3], [5, 5], [5, 6], [6, 7], [7, 7], [10,10],[10,11],[12,12],[13,13],[15,15],[16,16]])
    #print("w_matrix")
    np.set_printoptions(3)
    #print(w_matrix(sample,3))
    #Vc = k0graph(sample)
    Kc = 3
    #print("initial set", Vc)
    #print("affinity", Affinity(sample, Vc, Kc))
    #print("neighbor set", neighbor_set(sample, Vc, Kc))
    Vc = AGDL(sample,4, 3, 3)