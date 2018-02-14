import numpy as np
from scipy.io import loadmat
import neural_net as net
import matplotlib.pyplot as plt

D = 784 # input dimension
m = 9 # no of classes
lrate = 1e-3

labels, images, test_labels, test_images = net.load_data('assignment1/emnist-balanced.mat')
t_size = int(images.shape[0]*0.8)
X_train = images[:t_size, :]
Xu = np.mean(X_train, axis=0)
X_train = (X_train-Xu)/255.0
y_train = labels[:t_size, :]
X_test = images[t_size: , :]
Xu = np.mean(X_test, axis=0)
X_test = (X_test-Xu)/255.0  
y_test = labels[t_size: , :]  

neurons = [net.Layer(D ,m, 'softmax')]

lrange = [0.03, 0.1, 0.3, 1, 3, 10]
acc = []

for l in lrange:
	NN = net.NeuralNetwork(1, D, m, cost = 'crossent', layers = neurons, rate = lrate)
	NN.train(X_train, y_train, n_epoch = 10000, batch= 30, reg="l2", l=l, verbose = False)
	test_out = NN.predict(X_test)
	acc.append(net.accuracy(test_out, y_test))


plt.plot(lrange, acc)
plt.title("hyper-parameter tuning")
plt.xlabel("reg parameter")
plt.ylabel("accuracy %")
plt.show()
