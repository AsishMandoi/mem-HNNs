import tensorflow as tf
import numpy as np
import random

class HopfieldNetwork:
	def __init__(self, num_neurons):
		self.num_neurons = num_neurons
		self.weights = tf.Variable(np.zeros((num_neurons, num_neurons)), dtype=tf.float32)

	def train(self, patterns):
		patterns = np.array(patterns)
		self.weights.assign(np.matmul(patterns.T, patterns) / self.num_neurons)

	def recall(self, input_pattern, steps=5):
		input_pattern = np.array(input_pattern, dtype=np.float32)
		for _ in range(steps):
			output = tf.matmul(input_pattern, self.weights)
			output = tf.sign(output)
			input_pattern = output
		return output


class StochasticHopfieldNetwork:
	def __init__(self, num_neurons):
		self.num_neurons = num_neurons
		self.weights = tf.Variable(np.zeros((num_neurons, num_neurons)), dtype=tf.float32)

	def train(self, patterns):
		patterns = np.array(patterns)
		self.weights.assign(np.matmul(patterns.T, patterns) / self.num_neurons)

	def sigmoid(self, x):
		return 1 / (1 + tf.exp(-x))

	def stochastic_step(self, x):
		logits = tf.matmul(x, self.weights)
		prob = self.sigmoid(logits)
		output = tf.where(prob > tf.random.uniform(tf.shape(prob)), tf.ones_like(x), -tf.ones_like(x))
		return output

	def recall(self, input_pattern, steps=5):
		input_pattern = np.array(input_pattern, dtype=np.float32)
		for _ in range(steps):
			output = self.stochastic_step(input_pattern)
			input_pattern = output
		return output

class MaxCutHopfieldNetwork:
	def __init__(self, num_vertices):
		self.num_vertices = num_vertices
		self.weights = tf.Variable(np.zeros((num_vertices, num_vertices)), dtype=tf.float32)

	def train(self, adjacency_matrix):
		self.weights.assign(adjacency_matrix)

	def sigmoid(self, x):
		return 1.0 / (1.0 + tf.exp(-x))

	def stochastic_step(self, x, T=1.0):
		y_ = tf.matmul(x, self.weights)
		prob = self.sigmoid(y_ / T)
		output = tf.where(prob > tf.random.uniform(tf.shape(prob)), tf.ones_like(x), -tf.ones_like(x))
		return output

	def recall(self, input_pattern, steps=100):
		input_pattern = np.array(input_pattern, dtype=np.float32)
		print(input_pattern.shape)
		for _ in range(steps):
			output = self.stochastic_step(input_pattern)
			input_pattern = output
		return output

	def max_cut(self, adjacency_matrix, num_iterations=10):
		best_cut_edges = -1
		best_cut = None

		for _ in range(num_iterations):
			self.train(adjacency_matrix)
			random_initial_state = np.random.choice([-1, 1], size=(self.num_vertices,))
			cut_solution = self.recall(random_initial_state)
			cut_edges = np.sum(adjacency_matrix * (np.outer(cut_solution, -cut_solution) + np.eye(self.num_vertices)))
			
			if cut_edges > best_cut_edges:
				best_cut_edges = cut_edges
				best_cut = cut_solution

		return best_cut, best_cut_edges