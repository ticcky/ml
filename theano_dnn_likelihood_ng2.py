import theano
import theano.sandbox.linalg
from theano import function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import numpy.random as rng


# Make the random generator generate the same random numbers, every time we run.
np.random.seed(0)


# Training data for XOR.
data = [
    ([-1, -1], 0),
    ([1, 1], 0),
    ([-1, 1], 1),
    ([1, -1], 1),
]
data = data * 1000

data_x, data_y = zip(*data)


# Evaluate the trained model on the data and return the relative error.
def eval_model():
    err = 0
    for (_, t_y), p_y in zip(data, f_y(data_x)):
        res = p_y

        if res[0] > res[1]:
            if t_y != 0:
                err += 1
        else:
            if t_y != 1:
                err += 1

    return err * 1.0 / len(data)


# Parameters.
alpha = 1e-3  # Prevent fisher from being singular.
sigma = 0.1  # Training with additive noise variance.
rho = 0.5  # Dropout training binomial parameter.
n_iters = 300
dim_x = 2  # Dimensionality of the input.
dim_y = 2  # Dimensionality of the output.
n_hidden = [5, 7, 9]  # Number of layers.
activations = [T.tanh, T.tanh, T.tanh, T.nnet.softmax]  # NOTE: The last function goes to the output layer.
assert len(n_hidden) + 1 == len(activations)


params = []  # Keep model params here.


# Model definition.
X = T.matrix('X')  # Matrix with input data. Each row is 1 example.

# Build the layered neural network.
layers = [dim_x] + n_hidden + [dim_y]

Y = X
for i, (n1, n2, act) in enumerate(zip(layers[:-1], layers[1:], activations)):    # Iterate over pairs of adjacent layers.
    w = theano.shared(
                       np.asarray(rng.uniform(
                                              low=-np.sqrt(6. / (n1 + n2)),
                                              high=np.sqrt(6. / (n1 + n2)),
                                              size=(n2, n1)),
                                  dtype=theano.config.floatX),
                       'w%d' % i, borrow=True)
    b = theano.shared(np.zeros(n2, dtype=theano.config.floatX), 'b%d' % (i + 1))

    Y = (
            act(T.tensordot(
                    (Y),
                    (w),
                    [[1], [1]])
                + b
            )
    )

    params.extend([w, b,])


# Compile function for computing P(X|params).
f_y = function([X], Y)


# Define the loss function.
true_y = T.ivector('true_y')  # Multi-class labels for the input data
                              # (integers).
loss = -T.log(Y[T.arange(Y.shape[0]), true_y])  # Negative log-likelihood vector.
total_loss = T.sum(loss)  # Sum all elements of the loss vector to get the total
                          # negative log-likelihood.
f_loss = function([X, true_y], total_loss)

# Compute gradient for each training example.
grads_matrix, updates = theano.scan(
        lambda i, y : (
                T.concatenate([x.flatten() for x in T.grad(loss[i], params)]),
        ),  # Build a vector for each example, where the dimensions correspond to
           # the parameters of the whole model -- they are flattened
           # and concatenated.
        sequences=T.arange(loss.shape[0]),  # Iterates over this.
        non_sequences=[loss])  # Passes this as an additional argument.


# Define fisher matrix as a sum of gradient matrices defined for each example.
fisher_matrix = T.sum(T.tensordot(grads_matrix, grads_matrix, [[], []]),
                      axis=[0, 2])
# Add some diagonal elements to the fisher matrix to avoid singularity problems.
fisher_matrix += alpha * T.identity_like(fisher_matrix)

f_fisher_matrix = function([X, true_y], fisher_matrix, updates=updates)


# Total gradient is just sum of gradients at all training examples.
total_grad = T.sum(grads_matrix, axis=0)
f_total_grad = function([X, true_y], total_grad)


# Natural gradient is gradient times inverse of fisher matrix.
natural_grad = T.tensordot(total_grad, theano.sandbox.linalg.ops.matrix_inverse(fisher_matrix), [[0], [0]])
f_natural_grad = function([X, true_y], natural_grad)


def f_natural_grad_parsed(dx, dy):
    """Compute natural gradient for the whole data."""
    ng = f_natural_grad(dx, dy)
    res = []
    for param in params:
        shape = param.shape.eval()
        res.append(ng[:np.prod(shape)].reshape(shape))
        ng = ng[np.prod(shape):]

    return res


# Do batch-gradient descent to learn the parameters.
learning_rate = 0.1
for i in range(n_iters):
    total_loss = 0.0

    param_grads = f_natural_grad_parsed(data_x, data_y)
    for param, param_grad in zip(params, param_grads):
        param.set_value(param.get_value() - learning_rate * param_grad)

    print "loss(%.4f) rel_err(%.2f)" % (f_loss(data_x, data_y), eval_model(), )

