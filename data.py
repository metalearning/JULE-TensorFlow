import numpy as np
from scipy.misc import *


def load_coil20():
    data = np.zeros((72*20,128,128))
    labels = np.ones(72)

    for i in range(2,21):
        labels = np.concatenate((labels,np.full(51,i)),axis=0)

    for i in range(1,21):
        for j in range(72):
            index = (i-1)*72+j
            data[index]=imread("./dataset/coil-20/obj" + str(i) + "__" + str(j) + ".png","L")

    return data, labels


#data, labels = load_coil20()
