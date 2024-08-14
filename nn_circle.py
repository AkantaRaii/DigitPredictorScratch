import numpy as np
import pandas as pd
import math
np.random.seed(12)
# importing data
circles=pd.read_csv('circles.csv')

X=circles[['X1','X2']].values
X=np.array(X)
y=circles['Y'].values
y=np.array(y)

# neural network layer 
class layer:
    def __init__ (self,input_features,output):
        self.weights=np.random.randn(input_features,output)
        self.biases=np.zeros((output,1))
    def forward(self,X):
        self.output=np.dot(X,self.weights)+self.biases

# activation funcitions
class sigmoid:
    def forward(self,X):
        self.output=1/(1+np.exp(-X))
class relu:
    def forward(self,X):
        self.output=np.max(0,X)

def bceloss(predicted, output):
    loss = 0
    predicted=np.clip(predicted,1e-15,1-1e-15)
    loss = -np.mean(output * np.log(predicted) + (1 - output) * np.log(1 - predicted))
    return loss


layer1=layer(2,1)
layer1.forward(X)
print(layer1.output[:10])
sig=sigmoid()
sig.forward(layer1.output)
print(sig.output[:10])
print(bceloss(sig.output,y))