"""
    A Perceptron is an algorithm used for SUPERVISED LEARNING of binary classifiers. 
    Binary classifiers decide whether an input, usually represented by a series of 
    vectors, belongs to a specific class. In short, a perceptron is a single-layer 
    neural network. They consist of four main parts including input values, weights
    and bias, net sum, and an activation function. 
"""

import numpy as np

class Perceptron:
    def __init__(self, n_dim, learn_fact):
        # The knowlegde is saved in 'W' vector and the bias variable
        self.w = -1 + 2*np.random.rand(n_dim)  # W vector, its values is between -1, 1
        self.b = -1 + 2*np.random.rand()       # bias B, its value is between -1, 1
        self.learn_fact = learn_fact           # learn factor
   
    def predict(self, X):                       # prediction
        """
            Predict an estimate 'Y' from the actual parameters values of the vector 'W'
            and the bias variable 'B'.
            Return the estimated Y.   
        """
        p = X.shape[1]              # No. patterns (columns)
        y_estimada = np.zeros(p)    # create the 'Y' estimate 
        
        for i in range(p):          # for each pattern of 'X'
            # make the dot multiplication of X(i) and vector 'w'
            y_estimada[i] = np.dot(self.w, X[:,i]) + self.b 
           
            if y_estimada[i] >= 0:  # activation function (decides to wich class belongs: 1 or 0)
                y_estimada[i] = 1
            else:
                y_estimada[i] = 0
        return y_estimada  
    
    def fit(self, X, Y, epochs=50):             # training
        """
           This function trains the algorithm. Receive the input X and 
            the wanted values Y. The function is going to iterate  a 
            number of epochs.
        """
        p = X.shape[1]            # No. columns
        for _ in range(epochs):   # for each epoch
            # now loop throw the number of patterns
            for i in range(p):    # for each pattern
                y_est = self.predict(X[:, i].reshape(-1, 1))        # make the prediction, '1' or '0'
                self.w += self.learn_fact * (Y[i] - y_est)* X[:, i] # readjust the synhaptic weights in case the prediction is bad
                self.b += self.learn_fact * (Y[i] - y_est)          # readjust the bais in case the prediction is bad
    
    