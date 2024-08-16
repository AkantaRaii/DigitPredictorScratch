import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# imporing the trained model parameter
with open('parameter.pkl','rb') as f:
    w1,b1,w2,b2=pickle.load(f)

def relu(x):
    return np.maximum(0,x)
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
def forward(w1,b1,w2,b2,X):
    z1=w1.dot(X)+b1
    a1=relu(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return a2

def show_prediction(index,x,w1,b1,w2,b2):

    x=x[:,index,None]
    a2=forward(w1,b1,w2,b2,x)
    prediction=np.argmax(a2,axis=0)

    print(prediction)

    current_image = x.reshape((28, 28)) * 255

    # Display the image
    plt.imshow(current_image, cmap='gray', interpolation='nearest')  # Use cmap='gray' for grayscale
    plt.show()

digits = pd.read_csv('test_digit.csv')

# Convert data to a numpy array
digits = np.array(digits)
x=digits.T
show_prediction(100,x,w1,b1,w2,b2)
