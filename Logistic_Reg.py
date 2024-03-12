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
  #np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

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
print(f" the training shape is: {X_train.shape}")
print(f" the test shape is: {X_test.shape}")

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self):
    #self.lr = lr
    #self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x):

    ##### WRITE YOUR CODE HERE #####
    return np.hstack(((np.ones(x.shape[0])).reshape(x.shape[0],1),x))
    #### END CODE ####

  def sigmoid(self, x):
    ##### WRITE YOUR CODE HERE ####
    if x.shape[1]!=self.w.shape[0]:
      x=self.add_ones(x)
    z=x@self.w
    return np.divide(1, 1 + np.exp(-z))
    #### END CODE ####

  def cross_entropy(self, x, y_true):
    ##### WRITE YOUR CODE HERE #####
    y_pred = self.sigmoid(x)
    # print(y_pred.shape, y_true.shape)
    loss = np.divide(-np.sum(y_true * np.log(y_pred)+(1-y_true)* np.log(1-y_pred)),x.shape[0])
    return loss
    #### END CODE ####


  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    ##### WRITE YOUR CODE HERE #####
    #x = self.add_ones(x)
    print('predict_proba')
    proba = self.sigmoid(x)
    return proba
    #### END CODE ####

  def predict(self,x):

    ##### WRITE YOUR CODE HERE #####
    probas = self.predict_proba(x)
    output = (probas>=0.5).astype(int)#1 if probas>=.5 else 0 #(probas>=0.5).astype(int)
      #convert the probalities into 0 and 1 by using a treshold=0.5
    return output
    #### END CODE ####

  def fit(self,x,y,lr,n_epochs):

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
      # print(y.shape, y_pred.shape)
      grads=np.divide(-x.T @ (y-y_pred ),y.shape[0])


      #update rule
      self.w=self.w - lr * grads


      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)

     
      print(f'loss for epoch {epoch}  : {loss}')
     
  def accuracy(self,y_true, y_pred):

    y_true = y_true.reshape((-1,1))
    y_pred = y_pred.reshape((-1,1))
    ##### WRITE YOUR CODE HERE #####
    # correct = 0
    # total = len(y_true)
    # for true, pred in zip(y_true, y_pred):
    #   if true == pred:
    #     correct += 1
    #   acc= correct / total
    acc= np.sum(y_true == y_pred)/y_true.shape[0]
    return acc
    #### END CODE ####

#model = LogisticRegression()
#model.fit(X_train,y_train,lr,epoch)

#ypred_train = predict(X_train)
#acc = model.accuracy(y_train,ypred_train)
#print(f"The training accuracy is: {acc}")
#print(" ")

#ypred_test = model.predict(X_test)
#acc = model.accuracy(y_test,ypred_test)
#print(f"The test accuracy is: {acc}")