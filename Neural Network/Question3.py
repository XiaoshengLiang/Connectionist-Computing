
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize



# Create the train data and test data
def createData(dim,size):
    X = np.random.uniform(-1,1,size = (dim,size))
    y=[]
    sum = 0
    for line in X:
        y_item = []
        for num in line:
            sum = sum + num
        sin_sum = math.sin(sum)
        y_item.append(sin_sum)
        y.append(y_item) 
    return X, y



class Neural_Network(object):
    
    #Define parameters of network
    def __init__(self, Lambda=0):        
        self.inputLayerSize = 4
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5
        
        #Weights of each layer based on 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization (allow us to tune the relative cost )
        self.Lambda = Lambda
    
    #Forward propagation 
    def feedforward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    
    #Activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    
    #Cauculate the cost by given y and yHat
    def costFunction(self, X, y):
        self.yHat = self.feedforward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(sum(sum(self.W1**2))+sum(sum(self.W2**2)))
        return J
    
    
    #Backward propagation     
    #Derivative of cost function   
    def costFunctionGradient(self, X, y):
        self.yHat = self.feedforward(X)      
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidGradient(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidGradient(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1  
        return dJdW1, dJdW2
    
    def sigmoidGradient(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionGradient(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



#Train the network
class trainer(object):
    
    def __init__(self, N):
        self.N = N
    
    #Track the cost function values 
    def callbackF(self, params):
        self.N.setParams(params)
        self.trainJ.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, trainX, trainY, testX, testY):
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Store costs:
        self.trainJ = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',                                  args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res



NN = Neural_Network()


#Set the train data and test data
trainX, trainY = createData(40, 4)
testX, testY = createData(10, 4)


T = trainer(NN)
T.train(trainX, trainY, testX, testY)

print ("\n")
print ("testY obtained after trainning:")
print (NN.feedforward(testX))

print ("\n")
print ("testY in dataset:")
print (testY)



#Compare the cost between train data and test data
plt.plot(T.trainJ,'g', label="train")
plt.plot(T.testJ,'r',label="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()




