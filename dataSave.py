from loadData import *
from knn import *

data, labels, _ = load_MNIST_test()
#data, labels, _ = load_coil20()
#data, labels, _ = load_coil100()
distance1, indices1 = knn(1, data)
np.save('k_dist_1_MNIST.npy', distance1)
np.save('k_indices_1_MNIST.npy', indices1)

print("1")

distance2, indices2 = knn(10, data)
np.save('k_dist_10_MNIST.npy', distance2)
np.save('k_indices_10_MNIST.npy', indices2)
