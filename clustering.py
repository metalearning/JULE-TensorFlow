
import numpy as np
from loadData import *
from AGDL import *
from measure import *

#data, labels, CLUSTER_NUMBER = load_coil100()
#data, labels, CLUSTER_NUMBER = load_coil20()
data, labels, CLUSTER_NUMBER = load_MNIST_test()

Ks = 25
Kc = 15

cluster = AGDL(data, CLUSTER_NUMBER, Ks, Kc)



labels_pred = np.zeros(len(labels),dtype='i')

for i in range(len(cluster)):
    for j in range(len(cluster[i])):
        labels_pred[cluster[i][j]]=i
#print(labels_pred)


print("nmi : %f" % NMI(labels, labels_pred))
print("AC : %f" % ACC(labels, labels_pred))

















