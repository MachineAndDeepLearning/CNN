import numpy as np
from  scipy.io import loadmat
from sklearn.utils import shuffle


class MNISTData(object):
	def __init__(self):
		pass

	def loadFlatData(self):
		train = loadmat('Data/HouseNumbers/train_32x32.mat')
		test = loadmat('Data/HouseNumbers/test_32x32.mat')


		Xtrain = self.flatten(train['X'].astype(np.float32) / 255)
		Ytrain = train['y'].flatten() - 1

		Xtest = self.flatten(test['X'].astype(np.float32) / 255)
		Ytest = test['y'].flatten() - 1

		return Xtrain, Ytrain, Xtest, Ytest

	def flatten(self, X):
		N = X.shape[-1]
		flat = np.zeros((N, 3072))
		for i in range(N):
			flat[i] = X[:, :, :, i].reshape(3072)
		return flat

class HouseNumbersData(object):
	def __init__(self):
		pass



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


def error_rate(p, t):
	return np.mean(p != t)

def relu(a):
	return a * (a > 0)


