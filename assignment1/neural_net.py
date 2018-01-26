import numpy as np

class Layer:
	def __init__(self, units_prev, units, act = 'sigmoid')
		self.act = act         #weights, bias, z
		self.inp = np.zeros((units_prev,1))
		self.units = units
		self.units_in_prev = units_prev
		self.weights = np.random.randn(units_prev, units)   #Try xavier initialisation                    
		self.bias = np.random.randn(units,1)
		self.z = np.zeros((units, 1)) # z = (w^l)T a^l-1 + b^l
		self.gradW = None
		self.gradb = None

	def forward(self, inp):
		self.inp = inp
		self.z = np.matmul(self.weights.T, self.inp) + self.bias
		return self.activate(self.z)

	def backprop(self, delta1, rate, out): 
		delta = np.multiply(delta1, self.act_deriv(self.z, out)) # this is the real delta
		self.gradb = delta
		self.gradW = np.matmul(self.inp, self.delta.T)
		self.bias = self.bias - rate*self.gradb
		self.weights = self.weights - rate*self.gradW
		return np.matmul(self.weights, delta), inp  # this is delta1, passes onto next layer; NOT delta of the next layer

	def activate(self, x):
		if(self.act == "sigmoid"):
			return sigmoid(x)
		if(self.act == 'softmax'):
			return softmax(x)

	def act_deriv(self,x, out):
		if(self.act == 'sigmoid'):
			return sig_deriv(x, out)
		if(self.act == 'softmax'):
			return smax_deriv(x, out)

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
		#return np.multiply(softmax(x), (1 -softmax(x)))
		return np.multiply(out,(1-out)) 

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

	def costFunc(self, trueval, out):
		if (self.cost =="crossent")
			error = - np.sum(np.multiply(trueval, np.log(out)), 0)
			delta1 =  - np.divide(trueval,out) # delta1 is NOT delta of the last layer, that is calculated within the layer
		return error, delta1


	def backProp(self, trueval, out):
		_, delta1 = self.costFunc(trueval, out) # trueval is a onehot encoded vector
		o = out
		# delta1 = trueval # because out cancels when this is multiplied by derv. of sigmoid/softmax, correspondigly modify those functions
		for lr in reversed(self.layers):
			delta, o2 = lr.backprop(delta1, self.rate, o)
			delta1 = delta
			o = o2


## Hyper-parameters
D = 784 # input dimension
m = 9 # no of classes
alpha = 0.01

neurons = [Layer(D,1024, 'sigmoid'), Layer(1024,512, 'sigmoid'), Layer(512,256, 'sigmoid'), Layer(256, 100, 'sigmoid'), Layer(100,m, 'softmax')]
NN = NeuralNetwork(4, D, m, layers = neurons, rate = alpha)