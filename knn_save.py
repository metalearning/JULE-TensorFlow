from data import *
from knn import *

data, labels = load_MNIST_test()
distance1, indices1 = knn(1, data)
np.save('k0_dist.npy', distance1)
np.save('k0_indices.npy', indices1)

distance2, indices2 = knn(10, data)
np.save('k_dist.npy', distance2)
np.save('k_indices.npy', indices2)
