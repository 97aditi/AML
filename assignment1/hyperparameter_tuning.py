import numpy as np
from scipy.io import loadmat
import neural_net as net
import matplotlib.pyplot as plt

D = 784 # input dimension
m = 9 # no of classes
lrate = 1e-3

labels, images, test_labels, test_images = net.load_data('assignment1/emnist-balanced.mat')

neurons = [net.Layer(D ,m, 'softmax')]

lrange = [0.03, 0.1, 0.3, 1, 3, 10]

for l in lrange:
	NN = net.NeuralNetwork(1, D, m, cost = 'crossent', layers = neurons, rate = lrate)
	NN.train(images, labels, n_epoch = 10000, batch= 30, reg="l2", l=l, verbose = False)
	test_out = NN.predict(test_images)
	error = net.accuracy(test_out, test_labels)
print (error,"%")