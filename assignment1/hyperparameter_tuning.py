import numpy as np
from scipy.io import loadmat
import neural_net as net
import matplotlib.pyplot as plt

D = 784 # input dimension
m = 9 # no of classes
lrate = 1e-3

def KFoldCV(X, y, l, k=3):
	t_size = X.shape[0]
	tr_acc = 0
	tst_acc = 0
	for i in range(k):
		st = int(i*t_size/3.0)
		end = int((i+1)*t_size/3.0)
		X_train = np.concatenate((X[:st,:], X[end:,:]), axis=0)
		y_train = np.concatenate((y[:st,:], y[end:,:]), axis=0)
		X_test = X[st:end, :]
		y_test = y[st:end, :]
		neurons = [net.Layer(D, m, 'softmax')]
		NN = net.NeuralNetwork(1, D, m, cost = 'crossent', layers = neurons, rate = lrate)
		NN.train(X_train, y_train, n_epoch = 1000, batch= 20, verbose = False, reg="l2", l=l)
		out = NN.predict(X_train)
		tr_acc += net.accuracy(out, y_train)
		out = NN.predict(X_test)
		tst_acc += net.accuracy(out, y_test)
	tr_acc = tr_acc/3
	tst_acc = tst_acc/3

	return tr_acc, tst_acc


labels, images, test_labels, test_images = net.load_data('assignment1/emnist-balanced.mat')
Xu = np.mean(images, axis=0)
images = (images - Xu)/255.0

lrange = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
tst_acc = []
tr_acc = []
maxa = 0
i = 0

for l in lrange:
	tr, tst = KFoldCV(images, labels, l)
	if (tst>maxa):
		maxa=tst
		i=l
	tst_acc.append(tst)
	tr_acc.append(tr)

print ((i, maxa))
plt.plot(lrange, tst_acc, label='validation accuracy')
plt.plot(lrange, tr_acc, label='training accuracy')
plt.title("hyper-parameter tuning")
plt.xlabel("reg parameter")
plt.ylabel("accuracy %")
plt.legend()
plt.show()
