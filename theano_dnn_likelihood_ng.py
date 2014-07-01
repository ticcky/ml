#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from theano import function
from theano import tensor as T
import numpy as np
import numpy.random as rng

# Make the random generator generate the same random numbers, every time we run.
np.random.seed(0)

# Parameters.
n_iters = 50
dim_x = 2  # Dimensionality of the input.
dim_y = 2  # Dimensionality of the output.
n_hidden = [3, 3, 3]  # Number of layers.
activations = [T.tanh, T.tanh, T.tanh, T.nnet.softmax]  # NOTE: The last function goes to the output layer.

alpha = 1e-3
gamma = 1e-6
learning_rate = 0.1

assert len(n_hidden) + 1 == len(activations)

# Training data for XOR.
data = [
    ([-1, -1], 0),
    ([1, 1], 0),
    ([-1, 1], 1),
    ([1, -1], 1),
]


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
                      

# Model definition.
x = T.vector('x')
params = []  # Keep model params here.

# Build the layered neural network.
y = x
layers = [dim_x] + n_hidden + [dim_y]

# Iterate over pairs of adjacent layers.
for i, (n1, n2, act) in enumerate(zip(layers[:-1], layers[1:], activations)):  
    w = theano.shared(
                       np.asarray(rng.uniform(
                                              low=-np.sqrt(6. / (n1 + n2)),
                                              high=np.sqrt(6. / (n1 + n2)),
                                              size=(n2, n1)), 
                                  dtype=theano.config.floatX), 
                       'w%d' % i, borrow=True)
    b = theano.shared(np.zeros(n2), 'b%d' % (i + 1))
    params.append((w, b))
    
    y = act(T.dot(w, y) + b)    
   
f_y = function([x], y)

# Define the loss function.
y_real = T.iscalar('y_real')  # The desired output vector.
loss = -T.log(y[0][y_real])  # Negative log-likelihood.

# Add regularization.
l2 = 0
for w, b in params:
    l2 += (w**2).sum() + (b**2).sum()
loss += gamma * l2

f_loss = function([x, y_real], loss, allow_input_downcast=True)

# Derive the gradients for the parameters.
g_losses = []
f_g_losses = []
for w, b in params:
    g_loss = T.grad(loss, wrt=[w, b])    
    f_g_loss = function([x, y_real], g_loss)
    f_g_losses.append(f_g_loss)



# Do batch-gradient descent to learn the parameters.
for i in range(n_iters):
    total_loss = 0.0
    
    # Prepare accumulating variables for gradients of the parameters.
    gradients = []
    n_F = 0
    for w, b in params:
        w_s = w.shape.eval()
        b_s = b.shape.eval()
        tg_w = np.zeros(w_s, dtype=theano.config.floatX)
        tg_b = np.zeros(b_s, dtype=theano.config.floatX)
        gradients.append((tg_w, tg_b))

        n_F += w_s[0]*w_s[1] + b_s[0]

    t_cg = np.zeros(n_F, dtype=theano.config.floatX)
    t_F = np.zeros([n_F, n_F], dtype=theano.config.floatX)

    # Go through the data, compute gradient at each point and accumulate it.
    for t_x, t_y in data:    
        pred_y = f_y(t_x)        
        total_loss += f_loss(t_x, t_y)        

        cg = []
        for f_g_loss, (tg_w, tg_b) in zip(f_g_losses, gradients):                        
            g_w, g_b = f_g_loss(t_x, t_y)
            tg_w += g_w
            tg_b += g_b

            cg.append(g_w.flatten())
            cg.append(g_b)

        cg = np.concatenate(cg)
        t_cg += cg
        t_F += np.outer(cg, cg)

    t_cg /= len(data)
    t_F /= len(data)
    t_F += alpha*np.identity(n_F)

    t_ng = np.dot(np.linalg.pinv(t_F),t_cg)

#    print t_cg
#    print t_ng

    print "(%d)" % i, "%.4f" % total_loss, "err(%.2f)" % eval_model()  #, f_loss([1, 1], [1, 0])

    t_ng_i = 0
    # Update parameters.
    for (w, b), (tg_w, tg_b) in zip(params, gradients):

        tg_w_s = tg_w.shape
        tg_b_s = tg_b.shape

        #print tg_w_s, tg_b_s

        tng_w = t_ng[t_ng_i:t_ng_i+tg_w_s[0]*tg_w_s[1]]
        tng_w = tng_w.reshape(tg_w_s)
        t_ng_i = t_ng_i+tg_w_s[0]*tg_w_s[1]

        tng_b = t_ng[t_ng_i:t_ng_i+tg_b_s[0]]
        t_ng_i = t_ng_i+tg_b_s[0]

        w.set_value(w.get_value() - learning_rate * tng_w)
        b.set_value(b.get_value() - learning_rate * tng_b)
