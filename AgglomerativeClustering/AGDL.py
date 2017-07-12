import numpy as np
from knngraph import *

def Affinity(X,Vc,Kc,W):
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

def Affinity2(C1, C2, Kc, W):


    ij = np.ix_(C1, C2)
    ji = np.ix_(C2, C1)

    W_ij, W_ji = W[ij], W[ji]
    Ci, Cj = len(C1), len(C2)

    ones_i = np.ones((Ci, 1))
    affinity = np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i)
    #print(affinity)
    return affinity[0,0]

def Affinity3(C, Vc, Kc, W):
    nc = len(Vc)
    affinity = np.zeros([nc])


    for i in range(nc):

        ij = np.ix_(C, Vc[i])
        ji = np.ix_(Vc[i], C)

        W_ij, W_ji = W[ij], W[ji]
        Ci, Cj = len(C), len(Vc[i])

        ones_i = np.ones((Ci, 1))
        affinity[i] = np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i)

    return affinity[0]



def neighbor_set(X, Vc, Kc, W):
    Ns, As = [], []
    #time1 = time.time()
    #A = Affinity(X, Vc, Kc, W)
    #np.save('A_MNIST.npy',A)
    A = np.load('A_MNIST.npy')
    #print("Affinity time : ", time.time() - time1)
    for i in range(len(A)):
        As.append([x for x in sorted(list(A[i]))[-1 * Kc:] if x > 0])  #   np.sort(A[i])[-1 * Kc:].tolist()
        n = len(As[i])
        if n==0:
            Ns.append([])
        else:
            Ns.append(A[i].argsort()[-1*n:].tolist())

    return Ns,As


def AGDL(X, nt, Ks, Kc):
    Vc = k0graph(X)
    print("k0graph complete")

    #W = w_matrix(X, Ks)
    W = np.load("W_" + str(Ks) + "_MNIST.npy")
    print("neighbor")
    Ns, As = neighbor_set(X, Vc, Kc, W)
    nc = len(Vc)

    while nc > nt:

        max_affinity = 0
        for i in range(len(Ns)):
            if len(As[i])==0:
                continue
            aff = max(As[i])
            if aff > max_affinity:
                j = int(Ns[i][As[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            print("index alias")


        print(len(Vc[max_index2]) == 0, max_affinity)
        Vc[max_index1].extend(Vc[max_index2])
        Vc[max_index2] = []


        Ns[max_index2] = []
        As[max_index2] = []



        for i in range(len(Ns)):
            if max_index1 in Ns[i]:
                index = Ns[i].index(max_index1)
                As[i][index] = Affinity2(Vc[i],Vc[max_index1], Kc, W)

            if max_index2 in Ns[i] and max_index1 != i:
                index = Ns[i].index(max_index2)
                del Ns[i][index]
                del As[i][index]
                if max_index1 not in Ns[i] and max_index1 != i:
                    Ns[i].append(max_index1)
                    As[i].append(Affinity2(Vc[i],Vc[max_index1], Kc, W))

        Ns[max_index1].extend(Ns[max_index2])
        Ns[max_index1] = list(set(Ns[max_index1]))
        As[max_index1] = []

        if nc < 50:
            print(Ns[max_index1])
        # Fine the Kc-nearest clusters for Cab

        for i in range(len(Ns[max_index1])):

            index = Ns[max_index1][i]
            As[max_index1].append(Affinity2(Vc[index], Vc[max_index1], Kc, W))

        nc = nc - 1


    Cluster = []
    for i in range(len(Vc)):
        if len(Vc[i]) != 0:
            Cluster.append(Vc[i])
    #print("Cluster : ", Cluster)
    length = 0
    for i in range(len(Cluster)):
        length += len(Cluster[i])
    print("Data number, Cluster number : ", length, len(Cluster))

    return Cluster



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


