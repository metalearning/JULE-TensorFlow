from sklearn import metrics
import numpy as np
from data import *
from GDL import *

data, labels = load_coil20()
data = data.reshape(len(data), 128*128)
n_cluster = 20
Kc = 10

print("Clustering Start")
Vc = GDL(data, n_cluster, Kc)
labels_pred = np.zeros(72*20)

for i in range(len(Vc)):
    for j in range(len(Vc[i])):
        labels_pred[Vc[i][j]]=i
nmi = metrics.normalized_mutual_info_score(labels, labels_pred)

print("nmi : %f" % nmi)

















