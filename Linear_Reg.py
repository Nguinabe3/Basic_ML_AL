#importation of usefull librairies
import numpy as np # for numerical compution
import matplotlib.pyplot as plt # for plotting the graphics


class LinearRegression: #Define th class linear regression
  '''
  The goal of this class is to create a Linear Regression class,
  that we will use as our model to fit a linear regression given a data point
  '''
  def initialisation(self,n): # the function for initialise the matrix weidth
    return np.zeros((n,1))

  def make_prediction(self, X, W): # for making prediction using the data and the matrix weidth
   #assert X.ndim > 1
   #assert W.ndim > 1
   #assert X.shape[1]==W.shape[0]
   return X@W

  def mse(self, y_true, y_pred): # computing the mean square error using thr y true and y predicted
    return np.divide((np.sum(y_pred - y_true))**2, y_true.shape[0])

  def gradient(self,X, y, W): # To compute the gradiant of the loss with respect to weidth
    return np.divide(2*X.T@(self.make_prediction(X, W)-y),X.shape[0])

  def update(self, W,grad,lr): # for updating the the the weidth
    return W-lr*grad
  
  def plot(self,X, y, theta, number_epoc):
    """Plotting function for features and targets"""
    xtest = np.linspace(0, 1, 10).reshape(-1,1)
    ypred = self.make_prediction(xtest, theta).reshape(-1,1)
    plt.scatter(X, y, marker="+")
    plt.xlabel("feature")
    plt.ylabel("target")
    plt.plot(xtest, ypred, color="orange")
    plt.show()

  def plot_loss(self, losses):
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss curve")
    plt.show()

  def train_basch_descent(self, X, y, number_epoc): # For training the basch gradient descent
    lr = 0.1 # setup the learning rate
    d = X.shape[1] # taking the number of fatures
    W = self.initialisation(d) # Calling the function initialisation to initialise the matrix weidth
    store_loss = [] # a liste to store our loss
    for e in range(number_epoc): # Running the epoch
      y_pred = self.make_prediction(X,W) #Calling the make prediction function to predict, given the data and the matrix weidth
      loss = self.mse(y_pred,y) # Compute loss
      grad = self.gradient(X,y,W) # Compute the gradient of the loss
      W = self.update(W,grad,lr) # Uddating the weidth

      store_loss.append(loss)
      print(f"\nEpoch {e}, loss {loss}")
      self.plot(X,y,W,number_epoc)
      self.plot_loss(store_loss)