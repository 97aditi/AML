import numpy as np
import scipy.io as sio

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

	def costFunc(self, trueval, out, l=0, reg="none"):
		if (self.cost =="crossent")
		error = - np.sum(np.multiply(trueval, np.log(out)), 0)
		delta1 =  - np.divide(trueval,out) # delta1 is NOT delta of the last layer, that is calculated within the layer
		if(reg == 'l2')
			sum = 0
			for i in range(hidlayers):
				sum  = sum+np.sum(np.square(self.layers[i].weights))
			error = error + l*sum
		if(reg =='l1')
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

	def train(self):


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


## Hyper-parameters
D = 784 # input dimension
m = 9 # no of classes
alpha = 0.01

neurons = [Layer(D,1024, 'sigmoid'), Layer(1024,512, 'sigmoid'), Layer(512,256, 'sigmoid'), Layer(256, 100, 'sigmoid'), Layer(100,m, 'softmax')]
NN = NeuralNetwork(4, D, m, layers = neurons, rate = alpha)

train = sio.loadmat()

