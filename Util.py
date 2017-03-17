import numpy as np
from  scipy.io import loadmat
from sklearn.utils import shuffle


def init_weight_and_biases(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind
