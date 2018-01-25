import numpy as np

class Layer:
	def __init__(self, units, units_prev, act)
		self.act = act         #weights, bias, z
		self.inp = np.zeros((units_prev,1))
		self.units = units
		self.units_in_prev = units_prev
		self.weights = np.randn(units, units_prev)   #Try xavier initialisation                    
		self.bias = np.zeros((units,1))
		self.z = np.zeros((units, 1)) #z is the activation of neurons in the layer
		self.grad = None

	def forward(self, inp):
		self.inp = inp
		self.z = self.activate(np.matmul(self.weights, self.inp) + self.bias)
		return self.z

	def backprop(self, error):

		return delta

	def activate(self, x):
		if(self.act == "sigmoid")
			return sigmoid(x)

	def sigmoid(x):
		return np.divide(1,np.add(1,np.exp(x)))


	def sig_deriv(x):
		return np.multiply(sigmoid(x),(1-sigmoid(x)))

	def softmax(x):

class NeuralNetwork:
	def __init__(self, hidlayers, n_inputs, n_outputs, cost = 'crossent' , layers = None):
		self.hidlayers = hidlayers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.layers = layers

             
	def forwardpass(self, inputarr):
		out = inputarr
		for lr in self.layers:
			out = lr.forward(out)
		return out

	def costfunc(cost, expected):
		if (cost =="crossent")
			error = np.negative(np.sum(np.multiply(expected, np.log(self.layers[hidlayers-1].z))))


	def backprop(self, expected):	
		delta[hidlayers-1] = 		#last layer error						
		for i in range(hidlayers-2, 0, -1)
			delta[i] = np.multiply(np.matmul(np.transpose(self.layers[i+1].weights), delta[i+1]), deriv(act, self.layers[i].z))

## Hyper-parameters
D = 784 # input dimension
m = 9 # no of classes

neurons = [Layer(D,1024), Layer(1024,512), Layer(512,256), Layer(256, 100), Layer(100,m)]
NN = NeuralNetwork(4, D, m, layers = neurons)