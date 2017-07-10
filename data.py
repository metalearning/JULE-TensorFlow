import numpy as np
from scipy.misc import *
from tensorflow.examples.tutorials.mnist import input_data



def load_coil20():
    data = np.zeros((72*20,128,128))
    labels = np.ones(72)

    for i in range(2,21):
        labels = np.concatenate((labels,np.full(72,i)),axis=0)

    for i in range(1,21):
        for j in range(72):
            index = (i-1)*72+j
            data[index]=imread("./dataset/coil-20/obj" + str(i) + "__" + str(j) + ".png","L")
    print("number of data : ", len(labels))
    return data, labels

def load_MNIST_test():

    mnist = input_data.read_data_sets("dataset/MNIST/", one_hot=False)
    data = mnist.test.images
    labels = mnist.test.labels

    print(len(data), len(labels))
    return data, labels

def load_MNIST_train():
    pass



data, labels = load_MNIST_test()
