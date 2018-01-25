import numpy as np


class neural network:
	def_init_(self, hidlayers, no_inputs, no_outputs, inputarr, cost):
		self.hidlayers = hidlayers
		self.no_inputs = no_inputs
		self.no_outputs = no_outputs
		self.inputarr = inputarr
		self.layers = []

		# add layers , activation of last layer is softmax

             
	def forwardpass(self):
		self.layer[0].z = self.activate(np.add(np.matmul(self.layers[i].weights,inputarr),self.layers[i].b), self.layers[i].act)
		for i in range(1, hidlayers):
			self.layers[i].z = self.activate(np.add(np.matmul(self.layers[i].weights,self.layers[i-1].z),self.layers[i].b), self.layers[i].act)



	def costfunc(cost, expected):
		if (cost =="CrossEntropy")
			error = np.negative(np.sum(np.multiply(expected, np.log(self.layers[hidlayers-1].z))))


	def backprop(self, expected):	
		delta[hidlayers-1] = 		#last layer error						
		for i in range(hidlayers-2, 0, -1)
			delta[i] = np.multiply(np.matmul(np.transpose(self.layers[i+1].weights), delta[i+1]), deriv(act, self.layers[i].z))
			










	def activate(x, act):
		if(act == "sigmoid")
			return sigmoid(x)


	def sigmoid(x):
		return np.divide(1,np.add(1,np.exp(x)))


	def sig_deriv(x):
		return np.multiply(sigmoid(x),(1-sigmoid(x)))

	def softmax():





class layer:
	def_init_(self, act, units, units_prev)
		self.act = act         #weights, bias, z
		self.units = units
		self.units_in_prev = units_prev
		self.weights = np.randn((units, units_prev))   #Try xavier initialisation                    
		self.bias = np.zeros((units),1)
		self.z = np.zeros((units, 1)) #z is the activation of neurons in the layer