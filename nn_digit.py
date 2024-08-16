import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

#importing csv data 
digits=pd.read_csv('../data/train_digit.csv')

#converting data to numpy array
data=np.array(digits)

#getting shape of data (row, col)
m,n=data.shape

#shuffling the data
np.random.shuffle(data)

#spliting data
data_dev=data[:1000].T
y_dev=data_dev[0] 
X_dev=data_dev[1:n]

data_train=data[1000:m].T
y_train=data_train[0]
X_train=data_train[1:n]
X_train=X_train/255
m_train = data_train.shape[1]  # This will give you the number of training samples




#initialize parameters
def init_params():
    w1 = np.random.randn(10, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2
def relu(X):
    return np.maximum(0,X)
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1  # 10*42000
    a1 = relu(z1)         # 10*42000
    z2 = w2.dot(a1) + b2  # 10*42000
    a2 = softmax(z2)      # 10*42000
    return z1, a1, z2, a2

def relu_derivative(X):
    return (X > 0).astype(float)
def one_hot(Y):
    num_classes = Y.max() + 1
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1).reshape(-1, 1)

    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1).reshape(-1, 1)

    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1=w1-lr*dw1
    w2=w2-lr*dw2
    b1=b1-lr*db1
    b2=b2-lr*db2
    return w1,b1,w2,b2
def get_predictions(A2):
    print(A2.shape)
    print(np.argmax(A2,axis=0).max())
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size




#test

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i) 
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.1, 500)

with open('parameter.pkl','wb') as f:
    pickle.dump((W1,b1,W2,b2),f)

