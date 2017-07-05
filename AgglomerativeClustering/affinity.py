import numpy as np
from knn import *
from knngraph import *

sample = np.array([[1,1], [1,2], [2,1], [2,2], [3,4], [1,7], [5,5], [5,6], [6,7], [7,7]])

# input : Cluster sets
# output : Affinity matrix

W = w_matrix(sample, 3)

def Affinity(Vc):
    a = 1
    K = 4
    n = len(sample)
    nc = len(Vc)

    affinity = np.zeros([nc,nc])
    W = w_matrix(sample, 3)

    for i in range(nc):
        for j in range(i+1,nc):
            ij = np.ix_(Vc[i],Vc[j])
            ji = np.ix_(Vc[j],Vc[i])
            #print (ij)
            W_ij, W_ji = W[ij], W[ji]
            Ci, Cj = len(Vc[i]),len(Vc[j])

            ones_i = np.ones((Ci,1))
            affinity[i][j] = np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i)

    return affinity





class TestAffinity(TestCase):
    def test_affinity(self):
        W = w_matrix(sample, 3)
        self.assertEqual(len(W[0][W[0]>0]),3)

    def test_submatrix(self):
        W = np.arange(16).reshape(4,4)
        index = np.ix_([0,2],[1,3])
        self.assertTrue((W[index]==np.array([[1,3],[9,11]])).all())

    def test_max_affinity(self):
        Vc = k0graph(w_matrix(sample, 1))
        A = Affinity(Vc)
        m = np.unravel_index(np.argmax(A), A.shape)
        self.assertTupleEqual(m,(1,2))



if __name__ == "__main__":
    main()

