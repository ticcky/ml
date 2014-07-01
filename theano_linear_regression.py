import numpy as np
import matplotlib.pyplot as plt

import theano
from theano import function
from theano import pp
from theano import tensor as T
from theano.printing import min_informative_str

# Load training data.
train_x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
train_y = np.asarray([1.0, 1.1, 3.0, 2.0, 5.0])

# Parameters.
learning_rate = 0.1
n_dims = 1
n_steps = 10

# Define model.
x = T.matrix(name='x')  # Input matrix with examples.
w = theano.shared(value=np.zeros((n_dims, 1),  # Parameters of the model.
        dtype=theano.config.floatX),
        name='w', borrow=True)
f = function([x], T.dot(w, x))  # Linear regression.

# Define objective function.
y = T.vector(name='y')  # Output vector with y values.

# Define loss function.
loss = T.mean((T.dot(x, w).T - y) ** 2)

# Build the gradient descent algorithm.
g_loss = T.grad(loss, wrt=w)

train_model = function(inputs=[],
                       outputs=loss,
                       updates=[(w, w - learning_rate * g_loss)],
                       givens={
                           x: train_x,
                           y: train_y
                       })

# Execute the gradient descent algorithm.
for i in range(n_steps):
    print "cost", train_model()
   
# Plot results.
plt.figure()
plt.plot([a[0] for a in train_x], train_y, 'r')
plt.plot([a[0] for a in train_x], [f([xx])[0] for xx in train_x], 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.show()
