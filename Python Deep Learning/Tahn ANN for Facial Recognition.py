import numpy as np
import matplotlib.pyplot as plt

from utils import getBinaryData, sigmoid, sigmoid_cost, error_rate, relu
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self, M):
        self.M = M
        pass

    def fit(self, X, Y, learning_rate=5*10e-7, reg=1.0 , epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z = self.forward(X)

            # gradient descent step
            pY_Y = pY - Y
            self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg * self.W2)
            self.b2 -= learning_rate * ((pY_Y).sum() + reg * self.b2)

            dZ = np.outer(pY_Y, self.W2) * (1 - Z**2)
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(np.sum(dZ, axis=0) + reg*self.b1)

            if i % 20 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print(f"i: {i} \n cost: {c} \n error: {e}")
                if e < best_validation_error:
                    best_validation_error = e
        print(f"best_validation_error: {best_validation_error}")

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    model = ANN(100)
    model.fit(X, Y, show_fig=True)
    print(model.score(X, Y))


if __name__ == '__main__':
    main()


#best_validation_error: 0.044