import numpy as np
import networkx as nx

def plot_graph(adjacency_matrix, mylabels):
  rows, cols = np.where(adjacency_matrix == 1)
  edges = zip(rows.tolist(), cols.tolist())
  gr = nx.Graph()
  gr.add_edges_from(edges)
  nx.draw(gr, node_size=1000, labels=mylabels, with_labels=True)


class MaxCut:
  def energy(self, x, W):
    '''x: {-1, 1}^n, W: R^(n x n)'''
    return 0.25 * (np.sum(W) - x.T @ W @ x)
  
  def get_adj_matrix(self, path, zero_indexed=False, form='list', **kwargs):
    print(path)
    if form == 'matrix': return np.loadtxt(path, delimiter=",")
    data = np.loadtxt(path, skiprows=1)
    if not zero_indexed: data[:,:-1] -= 1
    num_vertices = int(np.max(data[:,:-1]) + 1)
    adjacency_matrix = np.zeros((num_vertices, num_vertices))
    for i, j, w_ij in data:
      # if adjacency_matrix[i, j] or adjacency_matrix[j, i]: print('Repetition: edge ({}, {})'.format(i, j))
      adjacency_matrix[int(i), int(j)] = w_ij
      adjacency_matrix[int(j), int(i)] = w_ij
    return adjacency_matrix

class MaxWeightedClique:
  def energy(self, x, W, b):
    '''x: {-1, 1}^n, W: R^(n x n), b: R^n'''
    ones = np.ones(W.shape[0])
    return 0.125 * ((x + 1).T @ (W.T @ ones - b) - ((x + 1).T @ W @ (x + 1)))
    # return 0.125 * ((x + 1).T @ (4 * theta) - ((x + 1).T @ W @ (x + 1)))
  
  # Get adjacency matrix from DIMACS format file. Read a line only if the first character of a line is 'e'.
  def get_adj_matrix(self, path, zero_indexed=False, weights=True, form='list', **kwargs):
    usecols = (1, 2, 3) if weights else (1, 2)
    data = np.loadtxt(path, comments=['c','p'], usecols=usecols)
    if not weights: data = np.hstack((data, np.ones((data.shape[0], 1))))
    if not zero_indexed: data[:,:-1] -= 1
    num_vertices = int(np.max(data[:,:-1]) + 1)
    adjacency_matrix = np.zeros((num_vertices, num_vertices))
    for i, j, w_ij in data:
      # if adjacency_matrix[i, j] or adjacency_matrix[j, i]: print('Repetition: edge ({}, {})'.format(i, j))
      adjacency_matrix[int(i), int(j)] = w_ij
      adjacency_matrix[int(j), int(i)] = w_ij
    return adjacency_matrix

class MaxWeightedIndependentSet:
  def energy(self, x, W, b):
    '''x: {-1, 1}^n, W: R^(n x n), b: R^n'''
    ones = np.ones(W.shape[0])
    return 0.125 * ((x + 1).T @ (W.T @ ones - b) - ((x + 1).T @ W @ (x + 1)))
  
class MinWeightedVertexCover:
  def energy(self, x, W, b):
    '''x: {-1, 1}^n, W: R^(n x n), b: R^n'''
    ones = np.ones(W.shape[0])
    return -0.125 * ((x + 1).T @ (W.T @ ones + b) + ((x - 1).T @ W @ (x - 1)))

class PermutationStates:
  '''Use the `next` method to get the next state from a random permutation of states from [0, N)'''
  def __init__(self, N):
    self.N = N
    self.curr_i = -1
    self.curr_permutation = None

  def next(self):
    self.curr_i = (self.curr_i + 1) % self.N
    if self.curr_i == 0 or self.curr_permutation is None:
      self.curr_permutation = np.random.permutation(np.arange(self.N))
    return self.curr_permutation[self.curr_i]