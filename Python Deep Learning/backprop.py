import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
	Z = 1/(1+np.exp(-X.dot(W1) - b1))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z

# Get the classification rate
def class_rate(Y, P):
	n_c = 0 # number correct
	n_t = 0 # number total
	for i in range(len(Y)):
		n_t +=1
		if Y[i] == P[i]:
			n_c += 1
	return n_c/n_t

def derivative_w2(Z, T, Y):
	N, K = T.shape
	M = Z.shape[1]

	# #slow
	# ret1 = np.zeros((M, K))
	# for n in range(N):
	# 	for m in range(M):
	# 		for k in range(K):
	# 			ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]
	# return ret1

	#faster
	# ret2 = np.zeros((M,K))
	# for n in range(N):
	# 	for k in range(K):
	# 		ret2[:, k] += (T[n,k] - Y[n,k])*Z[n,:]
	# assert(np.abs(ret1 - ret2).sum() < 10e-10)


	#even faster
	# ret3 = np.zeros((M,K))
	# for n in range(N):
	# 	ret3 += np.outer(Z[n], T[n] - Y[n])
	# assert(np.abs(ret2 - ret3).sum() < 10e-10)

	# #fastest
	# ret4 = Z.T.dot(T - Y)
	# assert(np.abs(ret3 - ret4).sum() < 10e-10)
	# return ret3

	#Therefore we want to return ret4
	return Z.T.dot(T-Y)


def derivative_b2(T,Y):
	return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	N, D = X.shape
	M, K = W2.shape

	#slow
	# ret1 = np.zeros((D, M))
	# for n in range(N):
	# 	for k in range(K):
	# 		for m in range(M):
	# 			for d in range(D):
	# 				ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1- Z(n,m)*X[n,d])
	# return ret1

	#fast
	dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
	return X.T.dot(dZ)



def derivative_b1(T, Y, W2, Z):
	return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def cost(T, Y):
	tot = T * np.log(Y)
	return tot.sum()

def main():
	# data simulation
	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # number of hidden layers
	K = 3 # number of classes

	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2,2])
	X3 = np.random.randn(Nclass, D) + np.array([-2,2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Y)

	T = np.zeros((N,K))
	for i in range(N):
		T[i, Y[i]] = 1

	#visualisation of the data (3 random clusters)
	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()

	#random initialisation of weights
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	costs = []
	for epoch in range(100000):
		output, hidden = forward(X, W1, b1, W2, b2)
		if epoch % 100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			r = class_rate(Y, P)
			print(f"cost: {c} /n classification_rate: {r}")
			costs.append(c)

		W2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
		b1 += learning_rate * derivative_b1(T, output, W2, hidden)

	plt.plot(costs)
	plt.show()




