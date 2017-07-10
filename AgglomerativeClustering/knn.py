# k-nearest neighbors algorithm
# input : set of samples X
# output : the array of length k
from unittest import main, TestCase
import numpy as np
from sklearn.neighbors import NearestNeighbors

def dist(i, j):
    return np.linalg.norm(i-j)

def knn(k, X):
    neigh = NearestNeighbors(k+1).fit(X)
    kneighbors = neigh.kneighbors(X,k+1,)
    distance = np.array(kneighbors[0][:,1:])
    indices = np.array(kneighbors[1][:,1:])
    return distance, indices

"""
def knn_distance(k, X):
    neigh = NearestNeighbors().fit(X)
    ans = neigh.kneighbors(X,k+1)[0]
    ans = np.array(ans)
    return ans
"""

class TestDist(TestCase):
    x = np.array([7, 7])
    y = np.array([3, 4])
    def test_dist_dist(self):
        self.assertEqual(dist(self.x,self.y),5)
    def test_dist_selfdist(self):
        self.assertTrue(dist(self.x,self.x)==0)

class TestKnn(TestCase):
    X = np.array([[1,1], [1,2], [3,4], [5,5], [5,6], [6,7], [7,7], [1,7]])
    def test_knn_one(self):
        ans = knn(1, self.X, self.X[0])
        self.assertEqual(ans, [1])

    def test_knn_two(self):
        ans = knn(3, self.X, self.X[5])
        self.assertCountEqual(ans, [3,4,6])

"""
    def test_knn_distance(self):
        ans = knn_distance(2, self.X)
"""


if __name__ == '__main__':
    main()
