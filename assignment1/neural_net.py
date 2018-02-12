import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

class Layer:
	def __init__(self, units_prev, units, act = 'sigmoid', train = True)
		self.act = act         #weights, bias, z
		self.inp = np.zeros((units_prev,1))
		self.units = units
		self.units_in_prev = units_prev
		self.weights = np.random.randn(units_prev, units)   #Try xavier initialisation                    
		self.bias = np.random.randn(units,1)
		self.z = np.zeros((units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradW = None
		self.gradb = None
		self.train = train 

	def forward(self, inp, dropout = True, prob):
		self.inp = inp
		self.z = np.matmul(self.weights.T, self.inp) + self.bias

		return self.activate(self.z) 


	def backprop(self, delta1, rate, out, reg="none", l=0): 
		if(self.act == 'softmax'):
			delta = np.matmul(self.smax_deriv(self.z, out), delta1)
	    else:
			delta = np.multiply(delta1, self.act_deriv(self.z, out)) # this is the real delta
		self.gradb = np.sum(delta,axis = 1)
		self.gradW = np.sum(np.matmul(self.inp, self.delta.T), axis=1)
		self.bias = self.bias - rate*self.gradb
		if(reg =='none')
			self.weights = self.weights - rate*self.gradW
		else if(reg == 'l2')
			self.weights = self.weights - rate*self.gradW - rate*np.mutliply(l,self.weights)
		else if(reg == 'l1')
			self.weights = self.weights - rate*self.gradW - rate*np.multiply(l, np.sign(self.weights))
		return np.matmul(self.weights, delta), inp  # this is delta1, passes onto next layer; NOT delta of the next layer

	def activate(self, x, alpha = 0):
		if(self.act == "sigmoid"):
			return sigmoid(x)
		if(self.act == 'softmax'):
			return softmax(x)
		if(self.act == 'tanh'):
			return tanh(x)
		if(self.act == 'l_relu'):
			return leaky_relu(x, alpha)

	def act_deriv(self,x, out):
		if(self.act == 'sigmoid'):
			return sig_deriv(x, out)
		if(self.act == 'softmax'):
			return smax_deriv(x, out)
		if(self.act == 'tanh'):
			return tanh_deriv(out)
		if(self.act == 'l_relu'):
			return leaky_relu_deriv(x, alpha)

	def sigmoid(x):
		return np.divide(1,np.add(1,np.exp(x)))


	def sig_deriv(x, out):
		#return np.multiply(sigmoid(x),(1-sigmoid(x)))
		return np.multiply(out,(1-out))

	def softmax(x):
		a0 = max(x, 0)
		den = np.sum(np.exp(x-a0),0)
		a = np.divide(np.exp(x-a0), den)
		return a

	# TODO: optimisation needed
	def smax_deriv(x, out):
		d = -np.matmul(out, out.T) + np.diag(np.sum(out, axis =1))
		return d

	def tanh(x):
		num = np.sum(1, -1*np.exp(-2*x))
		den = np.sum(1, np.exp(-2*x))
		return (num/den)

	def tanh_deriv(x, out):
		return (1 - (out*out))

	def leaky_relu(x, alpha):
		if (x >= 0):
			return x
		else return (self.alpha * x)

	def leaky_relu_deriv(x, alpha):
		if (x >= 0):
			return 1
		else return self.alpha


class Batchnorm(Layer):
	def __init__(self, units_prev)
		self.inp = np.zeros((units_prev,1))
		self.x_cap = np.zeros((units_prev,1))
		self.units = units_prev
		self.units_in_prev = units_prev
		self.gamma = np.random.randn(units,1);
		self.beta = np.random.randn();
		self.z = np.zeros((units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradbeta = None
		self.gradgamma = None
		self.mu = np.mean(self.inp, axis=0)
		self.epsilon = 1e-8
		self.var = np.var(self.inp, axis=0) + epsilon

	def forward(gamma, beta, inp):
		self.x_cap = np.divide(self.inp-mu, np.sqrt(var))
		self.z = gamma*x_cap + beta
		return self.z,mu,var,gamma,beta

    def deriv(Z, delta1):
    	m,D = Z.shape
    	Xmean = self.inp - self.mu;
    	inv_var = np.inverse(np.sqrt(self.var))
    	gradXcap = delta1*self.gamma
    	gradvar = np.sum(gradXcap*Xmean*(-0.5)*inv_var**3, axis=0)
    	gradmu = np.sum(gradXcap*(-1)*inv_var, axis=0)+ np.mean(-2*Xmean, axis=0)
		gradX = gradXnorm*inv_var+gradvar*2*Xmean/m+gradmu/m
    	return gradX

	def backprop(self, delta1, rate, out): 
		delta = deriv(self.z, delta1) # this is the real delta
		self.gradbeta = np.sum(delta,axis = 1)
		self.gradgamma = np.sum(np.matmul(x_cap, self.delta.T), axis=1)
		self.beta = self.beta - rate*self.gradbeta
		self.gamma = self.gamma - rate*self.gradgamma
		return np.matmul(self.weights, delta), inp



class NeuralNetwork:
	def __init__(self, hidlayers, n_inputs, n_outputs, cost = 'crossent' , layers = None, rate = 0.01):
		self.hidlayers = hidlayers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.layers = layers
		self.rate = rate
		self.cost = cost
             
	def forwardPass(self, inputarr):
		out = inputarr # should be D x N
		for lr in self.layers:
			out = lr.forward(out)
		return out

	def costFunc(self, trueval, out, l=0, reg="none"): # trueval should be (kxN), N = no. of samples in minibatch, k = no. of output units
		if (self.cost =="crossent"):
			error = - np.sum(np.multiply(trueval, np.log(out)))
			delta1 =  - np.divide(trueval,out) # delta1 is NOT delta of the last layer, that is calculated within the layer
		else if (self.cost == "mse"):
			error = np.sum((trueval - out)**2)/(2*trueval.shape[1])
			delta1 = -(trueval-out)/trueval.shape[1]

		if(reg == 'l2'):
			sum = 0
			for i in range(hidlayers):
				sum  = sum+np.sum(np.square(self.layers[i].weights))
			error = error + l*sum
		if(reg =='l1'):
			sum = 0
			for i in range(hidlayers):
				sum  = sum+np.sum(np.absolute(self.layers[i].weights))
			error = error + l*sum
    
    	return error, delta1

	def backProp(self, trueval, out, reg="none", l=0):
		_, delta1 = self.costFunc(trueval, out, l, reg) # trueval is a onehot encoded vector
		o = out
		# delta1 = trueval # because out cancels when this is multiplied by derv. of sigmoid/softmax, correspondigly modify those functions
		for lr in reversed(self.layers):
			delta, o2 = lr.backprop(delta1, self.rate, o, reg, l)
			delta1 = delta
			o = o2

	def train(self, X, y, batch = 64, n_epoch = 1000):
		iteration = 0
		while (iteration < n_epoch):
			idx = np.random.randint(X.shape[0], size = batch)
			input_X = X[idx, :]
			input_y = y[idx, :]
			outputs = self.forwardPass(input_X)
			self.backProp(input_y, outputs)
			iteration += 1 


class Dropout(Layer):
	def __init__(self, prob):
		self.prob = prob
		self.drop = np.zeros((units, 1))

	def forward(self, inp, prob):
		self.inp = inp
		self.z = np.matmul(self.weights.T, self.inp) + self.bias

		if (train = True):
			self.drop = np.random.binomial(1, prob, size = units.shape)
 			return np.multiply(self.activate(self.z), drop)
		else:
			return self.activate(self.z) * prob

	def backprop(self, delta1, rate, out, prob): 
		delta = np.multiply(delta1, self.act_deriv(self.z, out)) # this is the real delta
		delta = np.multiply(delta, self.drop) 
		self.gradb = np.sum(delta,axis = 1)
		self.gradW = np.sum(np.matmul(self.inp, self.delta.T), axis=1)
		self.bias = self.bias - rate*self.gradb
		self.weights = self.weights - rate*self.gradW
		return np.matmul(self.weights, delta), inp  # this is delta1, passes onto next layer; NOT delta of the next layer

def make_onehot(y):
	yoh = np.zeros(len(y),9)
	labelmap = {10:0, 13:1, 16:2, 17:3, 18:4, 19:5, 20:6, 23:7, 24:8}
	for i in rang(len(y)):
		yoh[i, labelmap[y[i,0]] ] = 1
	return yoh

if __name__ == '__main__':		
	## Hyper-parameters
	D = 784 # input dimension
	m = 9 # no of classes
	alpha = 0.01

	neurons = [Layer(D,512, 'sigmoid'), Layer(512,256, 'sigmoid'), Layer(256, 100, 'sigmoid'), Layer(100,m, 'softmax')]
	NN = NeuralNetwork(3, D, m, layers = neurons, rate = alpha)

	data = sio.loadmat('../../EMNIST/balanced')
	total_training_images = data['dataset'][0][0][0][0][0][0][:]
	total_training_labels = data['dataset'][0][0][0][0][0][1][:]
	ix = np.where(total_training_labels == 10 or total_training_labels == 13 or total_training_labels == 17 
		or total_training_labels == 18 or total_training_labels == 19 or total_training_labels == 24)
	labels = make_onehot(total_training_labels[ix])
	train_size = len(labels)
	#images = total_training_images[ix].reshape(train_size, 28, 28, 1)
	images = total_training_images[ix]

	total_testing_images = data['dataset'][0][0][1][0][0][0][:]
    total_testing_labels = data['dataset'][0][0][1][0][0][1][:]
    ix = np.where(total_testing_labels == 10 or total_testing_labels == 13 or total_testing_labels == 17 
		or total_testing_labels == 18 or total_testing_labels == 19 or total_testing_labels == 24)
    test_labels = make_onehot(total_testing_labels[ix])
    test_images = total_testing_images[ix]

    NN.train(images, labels)
    """
     TODO: 
      plot training curve, 
      plot CV curve, 
      final test accuracy, debugging 
    """