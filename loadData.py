import numpy as np
from scipy.misc import *
from tensorflow.examples.tutorials.mnist import input_data



def load_coil20():
    data = np.zeros((72*20,128,128))
    labels = np.zeros(72,dtype='i')
    n_cluster = 20

    for i in range(1,n_cluster):
        labels = np.concatenate((labels,np.full(72,i)),axis=0)

    for i in range(n_cluster):
        for j in range(72):
            index = i*72+j
            data[index]=imread("./dataset/coil-20/obj" + str(i+1) + "__" + str(j) + ".png","L")
    #print("number of data : ", len(labels))
    data = data.reshape(len(data), 128 * 128)
    return data, labels, n_cluster

def load_coil100():
    data = np.zeros((72*100,128,128))
    labels = np.zeros(72,dtype='i')
    n_cluster = 100

    for i in range(1,n_cluster):
        labels = np.concatenate((labels,np.full(72,i)),axis=0)

    for i in range(n_cluster):
        for j in range(72):
            index = i*72+j
            data[index]=imread("./dataset/coil-100/obj" + str(i+1) + "__" + str(j*5) + ".png","L")
    #print("number of data : ", len(labels))
    data = data.reshape(len(data), 128 * 128)
    return data, labels, n_cluster

def load_MNIST_test():
    mnist = input_data.read_data_sets("dataset/MNIST/", one_hot=False)
    data = mnist.test.images
    data = data.reshape(len(data), 28 * 28)
    labels = mnist.test.labels
    n_cluster = 10

    #print(len(data), len(labels))
    return data, labels, n_cluster

def load_MNIST_train():
    mnist = input_data.read_data_sets("dataset/MNIST/", one_hot=False)
    data = mnist.train.images
    data = data.reshape(len(data), 28 * 28)
    labels = mnist.train.labels
    n_cluster = 10

    # print(len(data), len(labels))
    return data, labels, n_cluster


