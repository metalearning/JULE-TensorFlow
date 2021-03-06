import numpy as np
from knngraph import *



# input : Cluster sets
# output : Affinity matrix

def Affinity(X,Vc,Kc):
    a = 1
    #n = len(X)
    nc = len(Vc)

    affinity = np.zeros([nc,nc])
    W = w_matrix(X, Kc)

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
        sample = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 4], [1, 7], [5, 5], [5, 6], [6, 7], [7, 7]])
        W = w_matrix(sample, 3)
        self.assertEqual(len(W[0][W[0]>0]),3)

    def test_submatrix(self):
        W = np.arange(16).reshape(4,4)
        index = np.ix_([0,2],[1,3])
        self.assertTrue((W[index]==np.array([[1,3],[9,11]])).all())

    def test_max_affinity(self):
        sample = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 4], [1, 7], [5, 5], [5, 6], [6, 7], [7, 7]])
        Vc = k0graph(w_matrix(sample, 1))
        A = Affinity(sample, Vc, 2)
        m = np.unravel_index(np.argmax(A), A.shape)
        self.assertTupleEqual(m,(0,1))



if __name__ == "__main__":
    main()

