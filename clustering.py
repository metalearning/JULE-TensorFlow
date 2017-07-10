
import numpy as np
from data import *
from AGDL import *
from measure import *

#data, labels = load_coil20()
#data = data.reshape(len(data), 128*128)
#n_cluster = 20
data, labels = load_MNIST_test()
data = data.reshape(len(data), 28*28)
n_cluster = 10
Kc = 5

#print("Clustering Start")
Vc = AGDL(data, n_cluster, Kc)

labels_pred = np.zeros(len(labels))

for i in range(len(Vc)):
    for j in range(len(Vc[i])):
        labels_pred[Vc[i][j]]=i
#print(labels_pred)


print("nmi : %f" % NMI(labels, labels_pred))

















