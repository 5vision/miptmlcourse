
import numpy as np

from sklearn.metrics import accuracy_score


class SoftmaxNetwork(object):
    """
    Output layers is softmax.
    Loss function is cross-entropy.
    Networks includes hidden layers with arbitrary activation functions.
    Targets: vector with number of classes dim=1
    """

    def __init__(self, layers):
        self.layers = layers
        # sense checks
        if len(layers) > 1:
            for i in xrange(len(layers) - 1):
                assert layers[i].hidden_dim == layers[i+1].input_dim, 'Inconsistent dimensions, layers {} and {}.'.\
                    format(i, i+1)
        assert type(self.layers[-1].act).__name__ == 'Softmax', 'Last layer should be softmax for this class.'
        self.n_params = sum([np.prod(x.w.shape) + np.prod(x.b.shape) for x in self.layers])
        self.flatten_params = np.zeros(shape=(self.n_params,), dtype=np.float32)

    def _forward_pass(self, x):
        """ Forward pass."""
        for layer in self.layers:
            x = layer.forward_pass(x)

    def _backward_pass(self, x, y_tar):
        """ Backward pass through neural network. Forward pass is needed to fill in intermediate quantities."""
        self._forward_pass(x)
        # speciic formulas for softmax delta
        delta = self.layers[-1].a
        delta[range(len(y_tar)), y_tar] -= 1.
        batch_size = float(delta.shape[0])
        w = self.layers[-1].w
        self.layers[-1].delta = delta
        self.layers[-1].db = np.sum(self.layers[-1].delta, axis=0, keepdims=True) / batch_size
        self.layers[-1].dw = (self.layers[-1].x.T).dot(self.layers[-1].delta) / batch_size
        # backpropagation for other layers
        for layer in reversed(self.layers[:-1]):
            layer.delta = layer.backward_pass(w, delta)
            w = layer.w
            delta = layer.delta

    def predict(self, x):
        """ Prediction on batch input of shape (n_batch, n_features)."""
        self._forward_pass(x)
        out = self.layers[-1].a
        return np.argmax(out, axis=1)

    def calc_accuracy(self, x, y_tar):
        y_pred = self.predict(x)
        return accuracy_score(y_true=y_tar, y_pred=y_pred)

    def update_weights(self, x, y_tar, eps):
        """ Gradient descent step."""
        self._backward_pass(x, y_tar)
        for layer in self.layers:
            layer.w -= eps * layer.dw
            layer.b -= eps * layer.db

    def get_flatten_params(self):
        """ Method needed for using CEM and random opt."""
        ix = 0
        for layer in self.layers:
            size_w = np.prod(layer.w.shape)
            self.flatten_params[ix: ix + size_w] = layer.w.flatten()
            ix += size_w
            size_b = np.prod(layer.b.shape)
            self.flatten_params[ix: ix + size_b] = layer.b.flatten()
            ix += size_b
        return self.flatten_params

    def set_params_from_flatten_array(self, params):
        """ Method needed for using CEM and random opt. """
        assert params.shape == (self.n_params), 'Incorrect length of flattened parameters array.'
        ix = 0
        for layer in self.layers:
            size_w = np.prod(layer.w.shape)
            layer.w = params[ix: ix + size_w].reshape(layer.w.shape)
            ix += size_w
            size_b = np.prod(layer.b.shape)
            layer.b = params[ix: ix + size_b].reshape(layer.b.shape)
            ix += size_b


class FullyConnectedLayer(object):
    """
    Parameters:
    act_func: activation function
    input_dim: dimensionality of input
    hidden_dim: dimensionality of output (=number of neurons in the layer)
    weights_initializer: function for initializing weights
    delta: derivative of loss function w.r.t. weighted input z
    z: weighted input of the layer
    a: activation of the layer
    x: input of the layer
    w: weights, derivative is dw
    b: biases, derivative is db
    """
    def __init__(self, input_dim, hidden_dim, act_func, w_initializer):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_initializer = w_initializer
        self.act = act_func
        self.z, self.a, self.dw, self.db, self.delta = None, None, None, None, None
        self._init_params()

    def _init_params(self):
        """ Parameters initializer."""
        self.w = self.w_initializer(self.input_dim, self.hidden_dim)
        self.b = np.zeros((1, self.hidden_dim))

    def forward_pass(self, x):
        """ Forward pass through layer."""
        # TO BE IMPLEMENTED
        #self.x = ...
        #self.z = ...
        #self.a = ...
        return self.a

    def backward_pass(self, w, delta):
        """ Backward pass through layer. 'w' and 'delta' are matrices of higher layer. """
        batch_size = float(delta.shape[0])
        # TO BE IMPLEMENTED
        #self.delta =  ...
        #self.db = ...
        #self.dw = ...



