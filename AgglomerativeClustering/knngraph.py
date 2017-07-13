from unittest import main, TestCase
import numpy as np
from knn import *
import time

sample = np.array([[1,1], [1,2], [2,1], [2,2], [3,4], [1,7], [5,5], [5,6], [6,7], [7,7]])

def w_matrix(X,Ks):
    a = 10
    n = len(X)

    weight_matrix = np.zeros([n, n])

    distance, indices = knn(Ks, X)
    #np.save('k_dist_' + str(Ks) + '_MNIST.npy',distance)
    #np.save('k_indices_' + str(Ks) + '_MNIST.npy',indices)
    #distance, indices = np.load('k_dist_' + str(Ks) + '_coil100.npy'), np.load('k_indices_' + str(Ks) + '_coil100.npy')
    #distance, indices = np.load('k_dist_' + str(Ks) + '_MNIST.npy'), np.load('k_indices_' + str(Ks) + '_MNIST.npy')
    #current1= time.time()
    #print("knn time: ", current1 - start)
    sigma2 = (a / n / Ks) * np.linalg.norm(distance)**2

    for i in range(n):
        #for j in range(n):
        for j in indices[i]:
            weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(X[i]- X[j])** 2) / sigma2 ** 2)
    np.save('W_' + str(Ks) + '_MNIST.npy',weight_matrix)
    return weight_matrix

def k0graph(X):
    #print("k0graph")


    W = w_matrix(X, 1)
    #W = np.load("W_1_MNIST.npy")

    #print("w_matrix")
    Vc = []
    n = len(W)

    """
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                for k in range(len(Vc)):
                    if j in Vc[k]:
                        Vc[k].append(i)
                        Vc[k] = list(set(Vc[k]))
                        j = n
                        break
                    if i in Vc[k]:
                        Vc[k].append(j)
                        Vc[k] = list(set(Vc[k]))
                        j = n
                        break
                else:
                    Vc.append([i,j])
    """
    x,y = np.where(W>0)

    for i in range(len(x)):
        for k in range(len(Vc)):
            if y[i] in Vc[k]:
                Vc[k].append(x[i])
                Vc[k] = list(set(Vc[k]))
                break
            if x[i] in Vc[k]:
                Vc[k].append(y[i])
                Vc[k] = list(set(Vc[k]))
                break
        else:
            Vc.append([x[i], y[i]])


    return Vc
    #np.set_printoptions(precision=3)
    #print(weight_matrix)


class TestK0graph(TestCase):
    def test_k0graph(self):
        Vc = k0graph(sample)
        self.assertCountEqual(Vc, [[0,1,2,3,4,5],[6,7],[8,9]])

if __name__ == "__main__":
    main()