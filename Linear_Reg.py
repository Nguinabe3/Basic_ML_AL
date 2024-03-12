import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

  def initialisation(self,n):
    return np.zeros((n,1))

  def make_prediction(self, X, W):
   return X@W

  def mse(self, y_true, y_pred):
    return np.divide((np.sum(y_pred - y_true))**2, y_true.shape[0])

  def gradient(self,X, y, W):
    return np.divide(2*X.T@(self.make_prediction(X, W)-y),X.shape[0])

  def update(self, W,grad,lr):
    return W-lr*grad
  def plot_loss(self, losses):
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss curve")
    plt.show()

  def train_basch_descent(self, X, y, number_epoc):
    #number_epoc=int(input("Enter the number of Epoch :"))
    lr = 0.1#float(input("Enter the learning :"))
    d = X.shape[1]
    W = self.initialisation(d)
    store_loss = []
    for e in range(number_epoc):
      y_pred = self.make_prediction(X,W)
      loss = self.mse(y_pred,y)
      grad = self.gradient(X,y,W)
      W = self.update(W,grad,lr)

      store_loss.append(loss)
      print(f"\nEpoch {e}, loss {loss}")
    return self.plot_loss(store_loss)