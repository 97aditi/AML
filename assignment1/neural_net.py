import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class Layer:
	def __init__(self, units_prev, units, act = 'sigmoid', alpha = 0, train = True):
		self.act = act         #weights, bias, z
		self.inp = np.zeros((units_prev,1))
		self.units = units
		self.units_in_prev = units_prev
		#xavier initialisation 
		self.weights = np.random.randn(units_prev, units).astype(np.float64)*np.sqrt(1.0/units_prev)                      
		self.bias = np.random.randn(units,1)
		self.z = np.zeros((units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradW = None
		self.gradb = None
		self.train = train 
		self.alpha = alpha

	def forward(self, inp, train= True):
		self.inp = inp
		self.z = np.matmul(self.weights.T, self.inp) + self.bias
		return self.activate(self.z)

	def backprop(self, delta1, rate, out, reg="none", l=0, cost = "crossent"): 
		if(self.act == 'softmax'):
			if (cost=="crossent"):
				delta = delta1
			else:
				delta = np.zeros(delta1.shape)
				smax_ = self.smax_deriv(self.z, out)
				i = 0
				for s in smax_:
					delta[:,i] = np.matmul(s, delta1[:,i])
					i+=1                                                                         
		else:
			delta = np.multiply(delta1, self.act_deriv(self.z, out)) # this is the real delta
		self.gradb = np.sum(delta,axis = 1).reshape(self.units,1)
		self.gradW = np.matmul(self.inp, delta.T)
		self.bias = self.bias - rate*self.gradb
		if(reg =='none'):
			self.weights = self.weights - rate*self.gradW
		elif(reg == 'l2'):
			self.weights = self.weights - rate*self.gradW - rate*l*self.weights
		elif(reg == 'l1'):
			self.weights = self.weights - rate*self.gradW - rate*l*np.sign(self.weights)
		return np.matmul(self.weights, delta), self.inp  # this is delta1, passes onto next layer; NOT delta of the next layer

	def activate(self, x):
		if(self.act == "sigmoid"):
			return self.sigmoid(x)
		if(self.act == 'softmax'):
			return self.softmax(x)
		if(self.act == 'tanh'):
			return self.tanh(x)
		if(self.act == 'l_relu'):
			return self.leaky_relu(x)

	def act_deriv(self,x, out):
		if(self.act == 'sigmoid'):
			return self.sig_deriv(x, out)
		if(self.act == 'softmax'):
			return self.smax_deriv(x, out)
		if(self.act == 'tanh'):
			return self.tanh_deriv(out)
		if(self.act == 'l_relu'):
			return self.leaky_relu_deriv(x)

	def sigmoid(self, x):
		x=np.clip(x,-500,500)
		return np.divide(1,np.add(1,np.exp(-x)))


	def sig_deriv(self,x, out):
		return np.multiply(out,(1-out))

	def softmax(self, x):
		a0 = np.max(x, 0)
		den = np.sum(np.exp(x-a0),0)
		a = np.divide(np.exp(x-a0), den)
		return a

	# TODO: optimisation needed
	def smax_deriv(self, x, out):
		d = []
		for i in range(out.shape[1]):
			out_ = out[:,i]
			temp = -np.matmul(out_,out_.T) + np.diag(out_)
			d.append(temp)
		return d

	def tanh(self, x):
		ones = np.ones((x.shape[0], x.shape[1]))
		num = ones - np.exp(-np.multiply(2,x))
		den = ones + np.exp(-np.multiply(2,x))
		return (num/den)

	def tanh_deriv(self, out):
		return (1 - (out**2))

	def leaky_relu(self, x):
		x = np.clip(x, -500,500)
		f = np.vectorize(lambda v: (v if v>0 else v*self.alpha))
		return f(x)

	def leaky_relu_deriv(self, x):
		f = np.vectorize(lambda v: (1 if v>0 else self.alpha))
		return f(x)


class Batchnorm(Layer):
	def __init__(self, units_prev, act='none'):
		self.act=act
		self.inp = np.zeros((units_prev,1))
		self.x_cap = np.zeros((units_prev,1))
		self.units = units_prev
		self.units_in_prev = units_prev
		self.gamma = np.random.randn(self.units,1);
		self.beta = np.random.randn(self.units,1);
		self.z = np.zeros((self.units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradbeta = None
		self.gradgamma = None
		self.epsilon = 1e-5
		

	def forward(self,inp, train = True):
		self.inp=inp
		self.mu = np.mean(self.inp, axis=1).reshape(self.units,1)
		self.var = np.var(self.inp, axis=1) + self.epsilon
		self.var=self.var.reshape(self.units,1)
		self.x_cap = np.divide(self.inp-self.mu, np.sqrt(self.var))
		self.z = np.multiply(self.gamma,self.x_cap) + self.beta
		return self.z

	def deriv(self, delta1):
		m,D = self.z.shape
		Xmean = self.inp - self.mu;
		inv_var = 1.0/(np.sqrt(self.var))
		gradXcap = np.multiply(self.gamma, delta1)
		gradvar = np.sum(gradXcap*Xmean*(-0.5)*(inv_var**3), axis=1).reshape(self.units,1)
		gradmu = np.sum(-gradXcap*inv_var, axis=1)+ np.mean(-2*Xmean, axis=1)
		gradmu = gradmu.reshape(self.units, 1)
		gradX = gradXcap*inv_var+gradvar*2*Xmean/m+gradmu/m
		gradX=np.clip(gradX, -500, 500)
		return gradX

	def backprop(self, delta1, rate, out, reg="none", l=0, cost = "crossent"): 
		delta = self.deriv(delta1) # this is the real delta
		self.gradbeta = np.sum(delta1,axis = 1).reshape(self.units,1)
		self.gradgamma = np.sum(np.multiply(self.x_cap,delta1), axis=1).reshape(self.units,1)
		self.beta = self.beta - rate*self.gradbeta
		self.gamma = self.gamma - rate*self.gradgamma
		return np.multiply(self.gamma, delta), self.inp



class NeuralNetwork:
	def __init__(self, n_layers, n_inputs, n_outputs, cost = 'crossent' , layers = None, rate = 0.01):
		self.n_layers = n_layers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.layers = layers
		self.rate = rate
		self.cost = cost
             
	def forwardPass(self, inputarr, train = True):
		out = inputarr # should be D x N
		for lr in self.layers:
			out = lr.forward(out, train= train)
		return out

	def costFunc(self, trueval, out, l=0, reg = "none"): 
		# trueval should be (kxN), N = no. of samples in minibatch, k = no. of output units
		n = trueval.shape[1]
		if (self.cost =="crossent"):
			error = - (np.sum(np.multiply(trueval, np.log(out+1e-20))))/n
			if(self.layers[self.n_layers-1].act == "softmax"):
				delta1 = (out - trueval)/n
				#print(delta1)
			else: 
				delta1 =  - np.divide(trueval,out+1e-20)/n 
				# delta1 is NOT delta of the last layer, that is calculated within the layer
		elif (self.cost == "mse"):
			error = np.sum((trueval - out)**2)/(2*n)
			delta1 = -(trueval-out)/n

		if(reg == 'l2'):
			sum = 0
			for i in range(self.n_layers):
				sum  = sum + np.sum(np.square(self.layers[i].weights))
		elif(reg =='l1'):
			sum = 0
			for i in range(self.n_layers):
				sum  = sum+np.sum(np.absolute(self.layers[i].weights))
			error = error + l*sum

		return (error, delta1)

	def backProp(self, trueval, out, reg = "none", l=0):
		_, delta1 = self.costFunc(trueval, out, l, reg) # trueval is a onehot encoded vector
		o = out
		for lr in reversed(self.layers):
			delta, o2 = lr.backprop(delta1, self.rate, o, reg = reg, l = l, cost = self.cost)
			delta1 = delta
			o = o2

	def train(self, X, y, batch = 32, n_epoch = 1000, l = 0, reg = "none", verbose=True):
		t_size = int(X.shape[0]*0.8)
		X_train = X[:t_size, :]
		Xu = np.mean(X_train, axis=0)
		X_train = (X_train-Xu)/255.0
		y_train = y[:t_size, :]
		X_test = X[t_size: , :]
		Xu = np.mean(X_test, axis=0)
		X_test = (X_test-Xu)/255.0  
		y_test = y[t_size: , :]  
		train_error = np.zeros(int(n_epoch/10))
		test_error = np.zeros(int(n_epoch/10))
		#train_acc = np.zeros(n_epoch)
		#test_acc = np.zeros(n_epoch)
		i = 0 
		while (i < n_epoch):
			idx = np.random.randint(t_size, size = batch)
			input_X = X_train[idx, :]
			input_y = y_train[idx, :]
			outputs = self.forwardPass(input_X.T, train = True)
			# print (outputs)
			self.backProp(input_y.T, outputs, reg=reg, l=l)
			if (i%10 == 0 and verbose):
				train_error[int(i/10)], _ = self.costFunc(input_y.T, outputs, 0, 'none')
				#train_acc[i] = accuracy(input_y, outputs.T)
				outputs_test = self.forwardPass(X_test.T, train = False)
				test_error[int(i/10)], _ = self.costFunc(y_test.T, outputs_test, 0, 'none')
				if (i%100==0): print(i, test_error[int(i/10)]) 
			i += 1 
		if verbose:	
			plt.plot(np.arange(1,train_error.shape[0]+1), train_error, label='training error')
			plt.plot(np.arange(1,test_error.shape[0]+1), test_error, label='validation error')
			plt.xlabel('no. of epochs (x10)')
			plt.ylabel('error')
			plt.legend()
			plt.show()


	def predict(self, inputarr):
		out = self.forwardPass(inputarr.T, train = False)
		return out.T

	
class Dropout(Layer):
	def __init__(self, prob, units_prev, units, act = 'sigmoid', alpha = 0):
		self.prob = prob
		self.drop = np.zeros((units, 1))
		self.act = act         #weights, bias, z
		self.inp = np.zeros((units_prev,1))
		self.units = units
		self.units_in_prev = units_prev
		self.weights = np.random.randn(units_prev, units).astype(np.float64)*np.sqrt(1.0/units_prev)                      
		self.bias = np.random.randn(units,1)
		self.z = np.zeros((units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradW = None
		self.gradb = None 
		self.alpha = alpha

	def forward(self, inp, train = True):
		self.inp = inp
		self.z = np.matmul(self.weights.T, self.inp) + self.bias

		if (train == True):
			self.drop = np.random.binomial(1, self.prob, size = self.units)
			self.drop = self.drop.reshape(self.drop.shape[0],1)
			return np.multiply(self.activate(self.z), self.drop)
		else:
			return self.activate(self.z) * (1-self.prob)

	def backprop(self, delta1, rate, out, reg = "none", l = 0, cost = 'crossent'): 
		delta = np.multiply(delta1, self.act_deriv(self.z, out)) # this is the real delta
		delta = np.multiply(delta, self.drop)
		self.gradb = np.sum(delta,axis = 1).reshape(self.units,1)
		self.gradW = np.matmul(self.inp, delta.T)
		self.bias = self.bias - rate*self.gradb
		
		self.weights = self.weights - rate*self.gradW
		return np.matmul(self.weights, delta), self.inp  
		# this is delta1, passes onto next layer; NOT delta of the next layer

def make_onehot(y):
	yoh = np.zeros((y.shape[0],9))
	labelmap = {10:0, 13:1, 16:2, 17:3, 18:4, 19:5, 20:6, 23:7, 24:8}
	for i in range(y.shape[0]):
		yoh[i, labelmap[y[i,0]] ] = 1
	return yoh

def load_data(path):
	data = loadmat(path)
	total_training_images = data['dataset'][0][0][0][0][0][0][:].astype(np.float32)
	total_training_labels = data['dataset'][0][0][0][0][0][1][:]
	classes = [10, 13, 16, 17, 18, 19, 20, 23, 24]  
	#A,D,G,H,I,J,K,N,O
	ix = np.array([], dtype=np.int64)
	for l in classes:
		ix = np.append(ix, np.where(total_training_labels==l)[0])
	np.random.shuffle(ix)
	labels = make_onehot(total_training_labels[ix])
	train_size = len(labels)
	#images = total_training_images[ix].reshape(train_size, 28, 28, 1)
	images = total_training_images[ix]

	total_testing_images = data['dataset'][0][0][1][0][0][0][:].astype(np.float32)
	total_testing_labels = data['dataset'][0][0][1][0][0][1][:]
	ix = np.array([], dtype=np.int64)
	for l in classes:
		ix = np.append(ix, np.where(total_testing_labels==l)[0])
	test_labels = make_onehot(total_testing_labels[ix])
	test_images = total_testing_images[ix]

	return (labels, images, test_labels,test_images)

def accuracy(result, truth):
	res = np.argmax(result,1)
	tru = np.argmax(truth, 1)
	ix = res == tru
	correct = np.sum(ix)
	total = res.shape[0]
	percent_acc = (correct*(1.0)/total)*100
	return percent_acc

def confusion_matrix(result, truth, n_classes):
	res = np.argmax(result,1)
	tru = np.argmax(truth, 1)
	confuse=np.zeros((n_classes, n_classes))
	for i in range(n_classes):
		ix = tru==i
		for j in range(n_classes):
			iy = res==j
			correct=np.multiply(ix,iy)
			confuse[i][j]=np.around(np.sum(correct)/400.0, decimals=2)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(np.array(confuse), cmap=plt.cm.jet, interpolation='nearest')
	width, height = confuse.shape
	for x in range(width):
		for y in range(height):
			ax.annotate(str(confuse[x][y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')
	ax.set_xlabel('Predicted Label', fontsize=12)
	ax.set_xticks([0,1,2,3,4,5,6,7,8])
	ax.set_xticklabels(['A','D','G','H','I','J','K','N','O'],verticalalignment='top')
	ax.set_ylabel('True Label', fontsize=12, rotation=90)
	ax.set_yticks([0,1,2,3,4,5,6,7,8])
	ax.set_yticklabels(['A','D','G','H','I','J','K','N','O'],rotation=90)
	plt.show()
	
def f1_score(result, truth):
	res = np.argmax(result, 1)
	tru = np.argmax(truth, 1)
	ix = tru == res
	ix = ix[ix == True]
	tp = len(ix)
	fp = len(res) - tp 
	fn = len(tru) - tp
	
	if tp > 0:
		precision = float(tp)/(tp + fp)
		recall = float(tp)/(tp + fn)
		return 2 * ((precision * recall)/(precision + recall))
	else:
		return 0
	
if __name__ == '__main__':		
	## Hyper-parameters
	D = 784 # input dimension
	m = 9 # no of classes
	lrate = 0.01

	neurons = [Layer(D, 512, 'l_relu',alpha=0.01),Layer(512, 256, 'l_relu',alpha=0.01),Layer(256 ,m, 'softmax')]
	NN = NeuralNetwork(3, D, m, cost = 'crossent', layers = neurons, rate = lrate)
	labels, images, test_labels, test_images = load_data('emnist-balanced.mat')
	NN.train(images, labels, n_epoch = 5000, batch= 64, reg="none", l=0.001)
	test_out = NN.predict(test_images)
	error = accuracy(test_out, test_labels)
	print (error,"%")
	confusion_matrix(test_out, test_labels, m)
	wt=np.zeros(0)
	for i in range(NN.n_layers):
		d=NN.layers[i].units*NN.layers[i].units_in_prev
		wt=np.append(wt,NN.layers[i].weights.reshape(d))
	plt.hist(wt,edgecolor='r',bins=50, range=[-0.2,0.2])
	plt.xlabel("Magnitude of weights")
	plt.ylabel("Frequency")
	plt.show()
	
