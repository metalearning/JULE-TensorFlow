from unittest import main, TestCase
import numpy as np
from knn import *

sample = np.array([[1,1], [1,2], [2,1], [2,2], [3,4], [1,7], [5,5], [5,6], [6,7], [7,7]])

def w_matrix(X,K):
    a = 1
    n = len(X)

    weight_matrix = np.zeros([n, n])

    sigma2 = 0
    for i in range(n):
        for j in knn_distance(K, X, X[i]):
            sigma2 += j ** 2
    sigma2 *= (a / n / K)

    for i in range(n):
        knn_index = knn(K, sample, sample[i])
        for j in range(n):
            if j in knn_index:
                weight_matrix[i][j] = np.exp(-1 * (dist(sample[i], sample[j]) ** 2) / sigma2 ** 2)

    return weight_matrix

def k0graph(W):

    Vc = []
    n = len(W)
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                for k in range(len(Vc)):
                    if j in Vc[k]:
                        Vc[k].append(i)
                        Vc[k] = list(set(Vc[k]))
                        j=n
                        break
                else:
                    Vc.append([i,j])

    return Vc
    #np.set_printoptions(precision=3)
    #print(weight_matrix)


class TestK0graph(TestCase):
    def test_k0graph(self):
        W = w_matrix(sample, 1)
        Vc = k0graph(W)
        self.assertCountEqual(Vc, [[0,1,2,3,4,5],[6,7],[8,9]])

if __name__ == "__main__":
    main()