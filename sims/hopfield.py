import numpy as np

class HopfieldNeuralNetwork:
	'''
		**A general Hopfield Neural Network**
		
		annealing can be operated in various modes:\n
			- Baseline
			- StochasticSimulatedAnnealing
			- ExponentialAnnealing
			- SemiExponentialAnnealing
			- ExponentialSmoothing
			- PowerSmoothing
			- Hybrid (Stochastic + Exponential Annealing)
	'''
	
	def __init__(self, weights, bias=None, mode='Baseline'):
		self.W = np.array(weights)		# weight matrix
		self.n_neurons = len(self.W)	# number of neurons
		if bias is None: bias = np.zeros(self.n_neurons)
		self.b = bias									# bias vector
		
		self.annealing_mode = mode
		self.W_avg = np.sum(self.W) / (self.n_neurons * (self.n_neurons-1))
		self.b_avg = np.sum(self.b) / self.n_neurons
		self.idx = None						# indices of neurons to be updated

	def train(self, weights=None, patterns=None):
		'''Uses Hebbian learning rule to train the network. Relevant for
		pattern recognition problems. Not required for optimization problems'''
		self.W = np.zeros((self.n_neurons, self.n_neurons))
		if weights is not None:
			self.W = np.array(weights)
		else:
			for pattern in patterns:
				self.W += np.outer(pattern, pattern)
			np.fill_diagonal(self.W, 0)

	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))
	
	def neuron_out(self, W, x, b):
		'''if sum_j(w_ij * x_j + b_i)=0, return x_i'''
		y = W @ x + b
		return np.where(y > 0, 1, np.where(y < 0, -1, x[self.idx]))
	
	def exponential_weight_step(self, x, t):
		'''`t`: inf -> 0'''
		if t < 0: raise ValueError('t must be in the range [0, inf)')
		W_new = self.W[self.idx] * (1 - np.exp(-1/t))
		b_new = self.b[self.idx] * (1 - np.exp(-1/t))
		return self.neuron_out(W_new, x, b_new)
	
	def linear_weight_step(self, x, t):
		'''`t`: 0 -> 1'''
		if t < 0 or t > 1: raise ValueError('t must be in the range [0, 1]')
		W_new = self.W[self.idx] * t
		b_new = self.b[self.idx] * t
		return self.neuron_out(W_new, x, b_new)
	
	def exponential_smoothing(self, x, alpha):
		'''`alpha`: inf -> 0'''
		if alpha < 0: raise ValueError('alpha must be in the range [0, inf)')
		if alpha == 0:
			W_new = self.W[self.idx]
			b_new = self.b[self.idx]
		else:
			W_new = self.W_avg + np.where(self.W[self.idx]>=self.W_avg,
				 															alpha/(np.exp(alpha/(self.W[self.idx]-self.W_avg)) - 1),
																		- alpha/(np.exp(alpha/(self.W_avg-self.W[self.idx])) - 1))
			b_new = self.b_avg + np.where(self.b[self.idx]>=self.b_avg,
				 															alpha/(np.exp(alpha/(self.b[self.idx]-self.b_avg)) - 1),
																		- alpha/(np.exp(alpha/(self.b_avg-self.b[self.idx])) - 1))
		return self.neuron_out(W_new, x, b_new)

	def power_smoothing(self, x, alpha):
		'''`alpha`: 0 -> 1'''
		if alpha < 0 or alpha > 1: raise ValueError('alpha must be in the range [0, 1]')
		W_new = self.W_avg + np.where(self.W[self.idx] >= self.W_avg, (self.W[self.idx] - self.W_avg)**alpha, -(self.W_avg - self.W[self.idx])**alpha)
		b_new = self.b_avg + np.where(self.b[self.idx] >= self.b_avg, (self.b[self.idx] - self.b_avg)**alpha, -(self.b_avg - self.b[self.idx])**alpha)
		return self.neuron_out(W_new, x, b_new)

	def stochastic_step(self, x, T):
		'''`T`: inf -> 0'''
		if T < 0: raise ValueError('T must be in the range [0, inf)')
		y_ = self.W[self.idx] @ x + self.b[self.idx]
		prob = self.sigmoid(y_ / T)
		
		# A vectorized function to probabilistically choose neuron outputs
		# depending on the individual sigmoidal probability at each neuron
		probabilistic_choice = np.vectorize(lambda z: np.random.choice([-1, 1], p=[1-z, z]))
		
		y = probabilistic_choice(prob)
		return y

	def hybrid_step(self, x, T, t):
		'''`T`: inf -> 0, `t`: inf -> 0'''
		if T < 0: raise ValueError('T must be in the range [0, inf)')
		if t < 0: raise ValueError('t must be in the range [0, inf)')
		y_ = self.W[self.idx] * (1 - np.exp(-1/t)) @ x + self.b[self.idx] * (1 - np.exp(-1/t))
		prob = self.sigmoid(y_ / T)
		
		# A vectorized function to probabilistically choose neuron outputs 
		# depending on the individual sigmoidal probability at each neuron
		probabilistic_choice = np.vectorize(lambda x: np.random.choice([-1, 1], p=[1-x, x]))
		
		y = probabilistic_choice(prob)
		return y
	
	def predict(self, x_pred, **kwargs):
		x_pred = np.array(x_pred, dtype=np.float32)
		self.idx = kwargs.get('indices', np.arange(self.n_neurons))
		if self.annealing_mode == 'Baseline':
			y_pred = self.neuron_out(self.W[self.idx], x_pred, self.b[self.idx])
		elif self.annealing_mode == 'StochasticSimulatedAnnealing':
			y_pred = self.stochastic_step(x_pred, kwargs['T'])
		elif self.annealing_mode == 'ExponentialAnnealing':
			y_pred = self.exponential_weight_step(x_pred, kwargs['t'])
		elif self.annealing_mode == 'SemiExponentialAnnealing':
			y_pred = self.linear_weight_step(x_pred, kwargs['t'])
		elif self.annealing_mode == 'ExponentialSmoothing':
			y_pred = self.exponential_smoothing(x_pred, kwargs['alpha'])
		elif self.annealing_mode == 'PowerSmoothing':
			y_pred = self.power_smoothing(x_pred, kwargs['alpha'])
		else:
			y_pred = self.hybrid_step(x_pred, kwargs['T'], kwargs['t'])
		return y_pred

	def energy(self, x):
		'''Hopfield energy to be minimized'''
		return -0.5 * x @ self.W @ x
