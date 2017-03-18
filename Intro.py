import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from Util import y2indicator, MNISTData, init_weight_and_biases, error_rate


def main():
	Xtrain, Ytrain, Xtest, Ytest = MNISTData().loadFlatData()

	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Xtest, Ytest = shuffle(Xtest, Ytest)

	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)


	max_iter = 20
	print_period = 10
	N, D = Xtrain.shape
	batch_sz = 500
	n_batches = N // batch_sz

	M1 = 1000
	M2 = 500
	K = 10
	W1_init, b1_init = init_weight_and_biases(D, M1)
	W2_init, b2_init = init_weight_and_biases(M1, M2)
	W3_init, b3_init = init_weight_and_biases(M2, K)

	# define tensorflow vars and expressions
	X = tf.placeholder(tf.float32, shape=[None, D], name='X')
	T = tf.placeholder(tf.float32, shape=[None, K], name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))

	Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
	Z2 = tf.nn.relu(tf.matmul(Z1,  W2) + b2)

	Yish = tf.matmul(Z2, W3) + b3
	cost =tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

	# used for error rate prediction
	predict_op = tf.argmax(Yish, 1)

	LL = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			for j in range(n_batches):
				Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
				Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

				session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
					prediction = session.run(predict_op, feed_dict={X: Xtest, T: Ytest_ind})
					err = error_rate(prediction, Ytest)

					print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
					LL.append(test_cost)

	plt.plot(LL)
	plt.show()

if __name__ == "__main__":
	main()