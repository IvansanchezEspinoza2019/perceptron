"""
    BODY MASS INDEX (BMI)
    This program uses the perceptron to predict if a certain given data 
    of people (its weight and height) has or no overweight.

    Features:
        1.- Weight is represented in kilograms(Kg).
        2.- Height is represented in meters(Mt).
    BmiDataSet class
        - Creates datasets with random weight and height of a given size, also 
          has a function to normalize the data. I utilize this class to create a 
          dataset for training the perceptron. but also this class can be util-
          ized to create test dataset (the data we want to predict).
    Steps:
        1.- Create a trainning dataset
        2.- Train the perceptron with that dataset(normalized)
        3.- Create test dataset(the data we want to predict)
        4.- Make predictions with the test dataset(also normalized)
"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

class BmiDataSet:
    def __init__(self, min_height=1.4, max_height=2.10, min_weight=40, max_weight=130, n_inputs=20):
        
        self.lim_inf = np.array([min_height, min_weight]) # weight and heigt lower limits
        self.lim_sup = np.array([max_height, max_weight]) # weight and heigt higher  limits
        self.sampleSize = n_inputs                        # sample size
        self.X = np.zeros((2, n_inputs))                  # X matrix 
        self.Y = np.zeros(n_inputs)                       # Y indicates which pattern has overweight(1) or not(0)
        
        ######## variables needed for data normalization #######
        self.x_min = np.zeros((2,1))                      # minimum values for weight and height
        self.x_max = np.zeros((2,1))                      # maximum values for weight and height
        
        self._initDataSet_()                              # initialize dataset (X, Y)
        
        ##### minimum and maximum vectors for weight and height  #####
        self.x_min = np.array([[np.min(self.X[0,: ])],    # min height in X
                               [np.min(self.X[1,: ])]])   # min weight in X 
        self.x_max = np.array([[np.max(self.X[0,: ])],    # max height in X
                               [np.max(self.X[1,: ])]])   # max weight in X
        
    def _initDataSet_(self):
        """
            Creates the dataset with random values. For each 
            pattern determines if has overweight or not.
        """
       
        for i in range(self.sampleSize):
            # create random values for weight and height
            self.X[:,i] = self.lim_inf + (self.lim_sup - self.lim_inf)* np.random.rand(2)
            # calculate the BMI
            imc = self.X[1, i] / self.X[0, i]**2 
            # overweight (1), not overweight (0)
            if imc <= 25:
                self.Y[i] = 0
            else:
                self.Y[i] = 1
                
    def _normalizar_(self, X):
        """
        Normalizes the dataset (rescales them). This is because 
        the values of the dataset should be between 0 and 1.
        Its very important that the dataset be normalized before be 
        processed by a learning algorthm.
        """
        ######### Min-Max normalization #########
        ## formula = (X - min) / (max - min)
        X = (X - self.x_min)/(self.x_max-self.x_min)
        
        return X
                
    def normalize(self, data=None):                  # receive the data
        if data is None:                             # normalize self.X if not data given
            return self._normalizar_(self.X)
        else:
            
            return self._normalizar_(data)
            """if self._isInRangeDataSet_(data):        # if the data should be in range of the dataset
                return self._normalizar_(data)          # normalize the data
            else:
                print("[Error: the data should be in range of the dataset ]")
                return -1
            """
        
    def _isInRangeDataSet_(self, X):
        """
            Verifies if a given data is in range of the dataset.
        """
        for i in range(X.shape[1]):
            if X[0, i] < self.x_min[0, 0] or X[1, i] < self.x_min[1, 0]:
                # if the weight o height is outside the dataset(smaller)
                return False
            elif X[0, i] > self.x_max[0,0] or X[1, i] > self.x_max[1,0]:
                # if the weight o height is outside the dataset(bigger)
                return False
        return True
    
def plot_dataset(X, Y):
    #### plot all data from a dataset ####
    plt_over = False
    plt_no_over = False
    
    for i in range(X.shape[1]):
        if Y[i] == 1:   # overweight
            if plt_over:
                plt.plot(X[0, i], X[1, i], 'ro')
            else:
                plt.plot(X[0, i], X[1, i], 'ro', label="[Train]Overweight")
                plt_over = True  
        else:         # no overweigt
            if plt_no_over:
                plt.plot(X[0, i], X[1, i], 'bo')
            else: 
                plt.plot(X[0, i], X[1, i], 'bo', label="[Train]No Overweight")
                plt_no_over = True
   
def plot_model(modelo):
    ### plot perceptron ####
    w1, w2, b = modelo.w[0], modelo.w[1], modelo.b
    li = 0
    ls = 1
    plt.plot([li, ls],[(1/w2)*(-w1*(li)-b), (1/w2)*(-w1*(ls)-b)],
             '-k', linewidth=1, label="Perceptron")
    
    



###### RANDOM DATASET FOR TRAININING #####
dataset= BmiDataSet(min_height=1.4, max_height=2.15, min_weight = 40, max_weight=130, n_inputs=150)

###### NORMALIZE TRAIN DATASET #####
train_x, train_y = dataset.normalize(), dataset.Y 

### OUR PERCEPTRON ####
neurona = Perceptron(2, 0.01) # (2, 0.01) beacause is 2 dimension problem and 0.01 because is our learn factor
print("Training sample size: "+str(train_x.shape[1]))

########## TRAINING #############
print("Training...")
neurona.fit(train_x, train_y, 120) #train

#### CREATE TEST DATASET  #######
test_data = BmiDataSet(n_inputs=20) # creates a new dataset but for testing

print("TEST DATA (ROW 0: HEIGHT, ROW 1: WEIGHT): \n\n"+str(test_data.X))
###### NORMALIZE TEST DATASET ######
test_normalized = dataset.normalize(test_data.X) # normalize with respect to the training dataset


################ PREDICTION ###############
prediction = neurona.predict(test_normalized)  # predict our normalized test dataset
print("\n-----------------------------------------------------------------")
print("NOTE: (0: NO overweight, 1: Overweight)")
print("PREDICTION: \n"+str(prediction))         # the prediction made by the perceptron
print("EXPECTED VALUES: \n"+str(test_data.Y))     # the real value of each pattern (which already been calculated when creating test dataset)
print("-------------------------------------------------------------------")


############### PLOT #################
print("\nPloting results...")

#### train dataset ####
plot_dataset(train_x, train_y)

### perceptron (represents a hyperplane) ####
plot_model(neurona)

### test data ####
plt.plot(test_normalized[0,: ], test_normalized[1,:], '*k',markersize=10,label="Test Data")

plt.title("BODY MASS INDEX (BMI)")
plt.xlabel(r'$HEIGHT (Mts)$')     # x axis
plt.ylabel(r'$WEIGHT (Kg)$')      # y axis
    

plt.grid()
plt.legend(loc='best')
plt.show()

