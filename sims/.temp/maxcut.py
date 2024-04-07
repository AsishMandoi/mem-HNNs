import utils
from hopfield import HopfieldNeuralNetwork
import numpy as np
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn-v0_8')

weights = utils.get_adj_matrix('max_cut_data/MaxCut_50.txt')
print(weights.shape)

hnn = HopfieldNeuralNetwork(weights=weights, mode='ExponentialAnnealing')
n_steps = 24
n_epochs = 1000

energies = np.zeros((n_epochs, n_steps))
cut_values = np.zeros((n_epochs, n_steps))

def cut(hnn, n_steps, epoch, batch_size=0.1):
  x = np.random.choice([-1, 1], size=hnn.n_neurons)
  for i, alpha in zip(range(n_steps), np.logspace(6, -5, n_steps)):
    y = hnn.predict(x, t=alpha)
    x = y
    cut_values[epoch][i] = np.sum(hnn.W * (np.ones((hnn.n_neurons, hnn.n_neurons)) - np.outer(y, y))) / 4
    energies[epoch][i] = hnn.energy(y)
    # print(f'Epoch: {epoch}, Step: {i}, Cut Value: {cut_values[epoch][i]},  Energy: {energies[epoch][i]}')
  cut_val = np.sum(hnn.W * (np.ones((hnn.n_neurons, hnn.n_neurons)) - np.outer(x, x))) / 4
  return cut_val, x

start = time.perf_counter()
ans = -1
for e in range(n_epochs):
  best_cut_val = -1e9
  best_cut = None
  
  cut_val, y = cut(hnn, n_steps, e)
  if cut_val > best_cut_val:
    best_cut_val = cut_val
    best_cut = y
  # print(f'Epoch: {e},\tBest Cut: {best_cut_val}')
  ans = max(ans, best_cut_val)

# plt.figure(1)
# for e in range(int(0.9*n_epochs), n_epochs): plt.plot(np.logspace(6, -5, n_steps), cut_values[e])
# plt.xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('cut value')
# plt.savefig('cut_values.png')

# plt.figure(2)
# for e in range(int(0.9*n_epochs), n_epochs): plt.plot(np.logspace(6, -5, n_steps), energies[e])
# plt.xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('energy')
# plt.savefig('energies.png')

print(f'Max Cut value: {ans}')
print(f'Finished in {time.perf_counter()-start:.2f} seconds')

(i, j) = np.unravel_index(np.argmax(energies, axis=None), energies.shape)
plt.figure(1)
plt.plot(np.logspace(6, -5, n_steps), energies[i])
plt.xscale('log')
plt.show()

(i, j) = np.unravel_index(np.argmax(cut_values, axis=None), cut_values.shape)
plt.figure(2)
plt.plot(np.logspace(6, -5, n_steps), cut_values[i])
plt.xscale('log')
plt.show()
