
import numpy as np
import matplotlib.pyplot as plt

def init_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) #/ np.sqrt(input_dim)

def plot_decision_boundary(x, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plt.cm.Spectral)


class CrossEntropyLoss(object):

    def loss(self, y, y_tar):
        return np.mean(np.nan_to_num(-y_tar * np.log(y)), axis=1)

    def loss_der(self, y, y_tar):
        return np.nan_to_num(-y_tar / y)


class Tanh(object):

    def act(self, x):
        return np.tanh(x)

    def act_der(self, x):
        return (1 - np.power(x, 2))


class Sigmoid(object):

    def act(self, x):
        return 1. / (1. + np.exp(-1. * x))

    def act_der(self, x):
        return x * (1 - x)


class Softmax(object):

    def act(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def act_der(self, x):
        raise NotImplementedError
