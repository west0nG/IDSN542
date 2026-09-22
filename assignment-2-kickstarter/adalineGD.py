import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
# initialize the AdalineGD object with the specified learning rate (eta), number of iterations (n_iter), and random seed (random_state)
    def fit(self, X, y):
# define a method called fit that takes in a 2D array X and a 1D array y as input
        rgen=np.random.RandomState(self.random_state)
# create a random number generator object (rgen) using the specified random seed (self.random_state) for reproducibility
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1])
# initialize weights to small random numbers from normal distribution with mean 0 and standard deviation 0.01 
# ...for each feature in the training set (X.shape[1] = number of features) 
        self.b_ = np.float64(0.)
# initialize bias to 0.0
        self.losses_ = []   
# initialize empty list to store loss values in each epoch
# at this point
# the adaline (self) will have .eta = 0.01 .n_iter = 50 .random_state = 1 .w_ = [...] .b_ =0.0 .losses_ =[]
        for i in range(self.n_iter):
# iterate over the number of epochs (n_iter)
# During each epoch, all training instances in X are used to update the weights once.
                net_input = self.net_input(X)
# use the net_input method (below) to calculate the net input (weighted sum of the inputs + bias) for each instance in X
# and store the resulting 1D array of net input values in the variable net_input
                output = self.activation(net_input)
# use the activation method to calculate the output of the Adaline model for each instance of net_input 
# and store the sequence of resulting values in the variable output 
# so output is a 1D array of predicted values (y^) for each training instance in X
# since Adaline uses the identity activation function the output is simply equal to the net input.
                errors = (y - output)
# calculate the prediction error for each traning instance 
# by subtracting the predicted output (y^) from the target label (y)
# and append the resulting sequence of errors to the variable errors
# The result is an array of errors (errors) that is then used to compute the gradient and update the weights.
                self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
# X.T -> transpose the input matrix (X) so that each row represents the value of one feature across all training instances 
# ....and each column represents one training instance
# X.T.dot(errors) performs a matrix-vector multiplication between the transposed input matrix and the errors vector
# ( in other words.... it computes the summation term in the partial derivative equation for each weight (wj) in the Adaline model )
# The result is a vector of gradients (one for each weight of each feature).
# eta -> scale each resulting value in the vector by the learning rate
# 2.0 -> and by multiplying by 2 that comes from differentiating the squared error
# X.shape[0] -> divide by the number of training instances to calculate the average gradient for each weight
# +=  -> add the calculated update to the each of the current weights
                self.b_ += self.eta * 2.0 * errors.mean()
# update the bias by adding the product of the learning rate (eta) * 2.0, * the mean of the errors
                loss = (errors**2).mean()
# calculate the mean squared error (loss) for this epoch by squaring all error values in the errors vector and taking the mean value of the resulting vector
                self.losses_.append(loss)
# append the mean squared error (loss) for this epoch to the losses_ list
        return self
# finished with training the Adaline object

    def net_input(self, X):
# define a new method called net_input
        return np.dot(X, self.w_) + self.b_
# that takes in a 2D array X and returns the net input (weighted sum of the inputs (x1*w1, x2*w2, ..., xn*wn) + bias) 
# for each instance in X (each row of the 2D array) - np.dot computes the dot product of X and the weights (w)
    def activation(self, net_input):
# define a new method called activation thtat takes in a 1D array of net input values
        return net_input
# and returns the input array (net_input) unchanged, as the activation function for Adaline is the identity function       
    def predict(self, X):
# define a new method called predict that takes in a 2D array X 
       return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
# use the net_input method to calculate the net input for each instance in X, 
# then use the activation method to calculate the output of the Adaline model for each instance of net_input
# if the output of the activation function is greater than or equal to 0.5, return 1, otherwise return 0
