import numpy as np
from knn import *

sample = np.array([[1,1], [1,2], [2,1], [2,2], [3,4], [1,7], [5,5], [5,6], [6,7], [7,7]])

def Affinity(Ci, Cj):
    a = 1
    n = len(sample)
    K = 4
    affinity = 0

    weight_matrix = np.zeros([n, n])


    sigma2 = 0
    for i in range(n):
        for j in knn_distance(K, sample, sample[i]):
            sigma2 += j**2
    sigma2 *= (a/n/K)

    for i in range(n):
        knn_index = knn(K,sample,sample[i])
        for j in range(n):
            if j in knn_index:
                weight_matrix[i][j]=np.exp(-1*(dist(sample[i],sample[j])**2)/sigma2**2)

    return affinity



class TestAffinity(TestCase):
    def test_affinity(self):


if __name__ == "__main__":
    main()