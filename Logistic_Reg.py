#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import make_classification
import sklearn
#Load the iris dataset from sklearn
X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)

def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test_split(X,y)
#print(f" the training shape is: {X_train.shape}")
#print(f" the test shape is: {X_test.shape}")

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self): # the init function to initialise our variables
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x): # To add the one column vector
    return np.hstack(((np.ones(x.shape[0])).reshape(x.shape[0],1),x))

  def sigmoid(self, x): # Implementation of sigmoid function for logistic regression
    if x.shape[1]!=self.w.shape[0]:
      x=self.add_ones(x)
    z=x@self.w
    return np.divide(1, 1 + np.exp(-z))

  def cross_entropy(self, x, y_true): # The cross entropy funtion to compute the loss
    y_pred = self.sigmoid(x)
    loss = np.divide(-np.sum(y_true * np.log(y_pred)+(1-y_true)* np.log(1-y_pred)),x.shape[0])
    return loss
  
  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    print('predict_proba')
    proba = self.sigmoid(x)
    return proba
  
  def predict(self,x): # this function use predict_proba function to make prediction
    probas = self.predict_proba(x)
    output = (probas>=0.5).astype(int) #convert the probalities into 0 and 1 by using a treshold=0.5
    return output

  def fit(self,x,y,lr,n_epochs): # this function is used to make prediction

    # Add ones to x
    x=self.add_ones(x)

    # reshape y if needed
    if y.shape[0]!=x.shape[0]:
      y=y.reshape((x.shape[0],1))
    # Initialize w to zeros vector >>> (x.shape[1])
    self.w=np.zeros((x.shape[1], 1))#.reshape((-1,1))

    for epoch in range(n_epochs):
      # make predictions
      y_pred=self.sigmoid(x).reshape((-1, 1))
      y = y.reshape((-1, 1))
      #compute the gradient
      grads=np.divide(-x.T @ (y-y_pred ),y.shape[0])
      #update rule
      self.w=self.w - lr * grads
      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)
      print(f'loss for epoch {epoch}  : {loss}')
      plt.plot(self.train_losses)
      plt.show()
     
  def accuracy(self,y_true, y_pred): # This function is used to compute the accuracy of the model
    y_true = y_true.reshape((-1,1))
    y_pred = y_pred.reshape((-1,1))
    acc= np.sum(y_true == y_pred)/y_true.shape[0]
    return acc