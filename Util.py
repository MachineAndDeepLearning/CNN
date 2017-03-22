import numpy as np
from  scipy.io import loadmat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


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


class FaceRecognizer(object):
	def __init__(self):
		self.label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	def getData(self, balance_ones=True):
		# images are 48x48 = 2304 size vectors
		# N = 35887
		X = []
		Y = []
		first = True

		for line in open('../Data/fer2013/fer2013.csv'):
			if first:
				first = False
			else:
				row = line.split(',')
				Y.append(int(row[0]))
				X.append([int(p) for p in row[1].split()])

		X, Y = np.array(X) / 255.0, np.array(Y)

		if balance_ones:
			# balance the 1 class
			X0, Y0 = X[Y != 1, :], Y[Y != 1]
			X1 = X[Y == 1, :]
			X1 = np.repeat(X1, 9, axis=0)
			X = np.vstack([X0, X1])
			Y = np.concatenate((Y0, [1] * len(X1)))

		return X, Y

	def getImageData(self):
		X, Y = self.getData()
		N, D = X.shape
		d = int(np.sqrt(D))
		X = X.reshape(N, 1, d, d)

		return X, Y

	def getBinaryData(self):
		X = []
		Y = []
		first = True
		for line in open('../Data/fer2013/fer2013.csv'):
			if first:
				first = False
			else:
				row = line.split(',')
				y = int(row[0])
				if y == 0 or y == 1:
					Y.append(y)
					X.append([int(p) for p in row[1].split()])
		return np.array(X) / 255.0, np.array(Y)

	def showImages(self, balance_ones=True):
		X, Y = self.getData(balance_ones=balance_ones)

		while True:
			for i in range(7):
				x, y = X[Y == i], Y[Y == i]
				N = len(y)
				j = np.random.choice(N)

				plt.imshow(x[j].reshape(48, 48), cmap='gray')
				plt.title(self.label_map[y[j]])
				plt.show()
			prompt = input('Quit? Enter Y:\n')
			if prompt == 'Y':
				break;


def init_weight_and_biases(M1, M2):
	M1 = int(M1)
	M2 = int(M2)
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


def error_rate(targets, predictions):
	return np.mean(targets != predictions)


def relu(a):
	return a * (a > 0)


def sigmoid(a):
	return 1 / (1 + np.exp(-a))


def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)


def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(poolsz)))
	return w.astype(np.float32)
