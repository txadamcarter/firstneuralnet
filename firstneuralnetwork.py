#firstneuralnetwork

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special

# neural network class definition
class neuralNetwork:

	# initialize the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden and output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# learning rate
		self.lr = learningrate

		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

		# create initial link weight matrices, w_ih (weight_input-->hidden) and w_ho (weight_hidden-->output)
		# weights inside the arrays are w_i_j, where link is from node i to node j in next layer
		# w11 w22
		# w12 w22 etc

		# the following two functions would supply an array of size (hidden X input) and 
		# (output X hidden). They use numpy's random number generator function, however. 

		#self.w_ih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		#self.w_ho = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

		# A slightly better approach is to initialize the weight matrices with data sampled 
		# from a normal distribution.

		self.w_ih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.w_ho = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		pass

	# train the neural network
	def train():
		pass

	# query the neural network
	def query(self, inputs_list):

		# convert linear inputs list to 2D array
		inputs = numpy.array(inputs_list, ndmin=2).T

		# calculate the signals into the hidden layer
		hidden_inputs = numpy.dot(self.w_ih, inputs)

		# calculate the signals emerging from the hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate the signals into the final output layer
		final_inputs = numpy.dot(self.w_ho, hidden_outputs)

		#calculate the final output
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
		pass

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

 