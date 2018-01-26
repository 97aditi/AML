import numpy as np

class Layer:
	def __init__(self, units_prev, units, act = "sigmoid"):
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

	def backprop(self, error, rate, reg="none", l=0):
		self.gradb = error
		self.gradW = np.matmul(self.inp, self.z.T)
		self.bias = self.bias - rate*self.gradb
		if(reg =='none')
			self.weights = self.weights - rate*self.gradW
		else if(reg == 'l2')
			self.weights = self.weights - rate*self.gradW - rate*np.mutliply(l,self.weights)
		else if(reg == 'l1')
			self.weights = self.weights - rate*self.gradW - rate*np.multiply(l, np.sign(self.weights))
		delta = np.multiply(np.matmul(self.weights, error), self.act_deriv(self.z))
		return delta

	def activate(self, x):
		if(self.act == "sigmoid"):
			return sigmoid(x)
		if(self.act == 'softmax'):
			return softmax(x)

	def act_deriv(self,x):
		if(self.act == 'sigmoid'):
			return sig_deriv(x)
		if(self.act == 'softmax'):
			return smax_deriv(x)

	def sigmoid(x):
		return np.divide(1,np.add(1,np.exp(x)))


	def sig_deriv(x):
		return np.multiply(sigmoid(x),(1-sigmoid(x)))

	def softmax(x):
		a0 = max(x, 0)
		den = np.sum(np.exp(x-a0),0)
		a = np.divide(np.exp(x-a0), den)
		return a

class NeuralNetwork:
	def __init__(self, hidlayers, n_inputs, n_outputs, cost = "crossent" , layers = None, rate = 0.01):  #l is the regularisation parameter
		self.hidlayers = hidlayers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.layers = layers
		self.l = l
		self.rate = rate
		self.cost = cost
             
	def forwardPass(self, inputarr):
		out = inputarr # should be D x N
		for lr in self.layers:
			out = lr.forward(out)
		return out

	def costFunc(self, trueval, l=0, reg="none"):
		if (self.cost =="crossent")
			if(reg == "none")
				error = np.negative(np.sum(np.multiply(trueval, np.log(self.layers[hidlayers-1].z))))
			if(reg == 'l2')
				sum = 0
				for i in range(hidlayers):
					sum  = sum+np.sum(np.square(self.layers[i].weights))
				error = np.negative(np.sum(np.multiply(trueval, np.log(self.layers[hidlayers-1].z)))) + l*sum
			if(reg =='l1')
				sum = 0
				for i in range(hidlayers):
					sum  = sum+np.sum(np.absolute(self.layers[i].weights))
				error = np.negative(np.sum(np.multiply(trueval, np.log(self.layers[hidlayers-1].z)))) + l*sum



	def backProp(self, trueval, reg="none", l=0):
		error = self.costFunc(trueval, l, reg)
		for lr in reversed(layers):
			delta = lr.backprop(error, self.rate, reg, l)
			error = delta


## Hyper-parameters
D = 784 # input dimension
m = 9 # no of classes
alpha = 0.01

neurons = [Layer(D,1024, 'sigmoid'), Layer(1024,512, 'sigmoid'), Layer(512,256, 'sigmoid'), Layer(256, 100, 'sigmoid'), Layer(100,m, 'softmax')]
NN = NeuralNetwork(4, D, m, layers = neurons, rate = alpha)