import theano
import theano.sandbox.linalg
from theano import function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import numpy.random as rng

trng = RandomStreams()

# Make the random generator generate the same random numbers, every time we run.
np.random.seed(0)


# Training data for XOR.
data = [
    ([-1, -1], 0),
    ([1, 1], 0),
    ([-1, 1], 1),
    ([1, -1], 1),
]

data_x, data_y = zip(*data)

print data_x


# Evaluate the trained model on the data and return the relative error.
def eval_model():
    err = 0
    for t_x, t_y in data:
        res = f_y(t_x)[0]
        
        if res[0] > res[1]:
            if t_y != 0:
                err += 1
        else:
            if t_y != 1:
                err += 1
    
    return err * 1.0 / len(data)
                      

# Parameters.
alpha = 1e-3  # Preven fisher being singular.
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
X = T.matrix('X')  # Each row is 1 example.

# Build the layered neural network.
layers = [dim_x] + n_hidden + [dim_y]

Y = (X)
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
    
lin_params = T.concatenate([p.flatten() for p in params])
   
f_y = function([X], Y)


#f_y(data_x)

#print 'ok'

# Define the loss function.
true_y = T.ivector('true_y')  # The desired output vector.

loss = (-T.log(Y[T.arange(Y.shape[0]), true_y]))  # Negative log-likelihood.
total_loss = T.sum(loss)

grads_matrix, updates = theano.scan(
        lambda i, y : T.concatenate([x.flatten() for x in T.grad(loss[i], params)]), 
        sequences=T.arange(loss.shape[0]), 
        non_sequences=[loss])

fisher_matrix = T.sum(T.tensordot(grads_matrix, grads_matrix, [[], []]), axis=[0, 2])
fisher_matrix += alpha * T.identity_like(fisher_matrix) 
f_fisher_matrix = function([X, true_y], fisher_matrix, updates=updates)

total_grad = T.sum(grads_matrix, axis=0)
f_total_grad = function([X, true_y], total_grad)
#print f_total_grad(data_x, data_y)

natural_grad = T.tensordot(theano.sandbox.linalg.ops.matrix_inverse(fisher_matrix), total_grad, [[0], [0]])
f_natural_grad = function([X, true_y], natural_grad)


def f_natural_grad_parsed(dx, dy):
    ng = f_natural_grad(dx, dy)
    res = []
    for param in params:
        shape = param.shape.eval()
        res.append(ng[:np.prod(shape)].reshape(shape))
    
    return res

#print f_natural_grad_parsed(data_x, data_y)
#raise Exception()

#f_grads_matrix = function([X, true_y], grads_matrix, updates=updates)
#fisher_matrix = T.sum(T.grad(loss, wrt=params))
#tst = f_fisher_matrix(data_x, data_y)
#print 'sasdf'
#print tst.shape
#print tst
#print len(tst), len(params)
#print tst[0,0,0,0]



f_loss = function([X, true_y], total_loss)

# Add regularization.
#gamma = theano.shared(1e-4)
#l2 = 0
#for param in params:
#    l2 += (param**2).sum()
#loss += gamma * l2

#f_loss = function([x, y_real], loss, allow_input_downcast=True)

#total_grad = T.concatenate(T.grad(loss, wrt=params))
#f_total_grad = function([X, true_y], total_grad)

    
# Do batch-gradient descent to learn the parameters.
learning_rate = 0.1
for i in range(n_iters):    
    total_loss = 0.0
    
    param_grads = f_natural_grad_parsed(data_x, data_y)    
    for param, param_grad in zip(params, param_grads):        
        param.set_value(param.get_value() - learning_rate * param_grad)
    
    print "%.4f" % f_loss(data_x, data_y)
    
