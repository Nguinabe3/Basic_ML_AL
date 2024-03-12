import numpy as np
import matplotlib.pyplot as plt
from Linear_Reg import LinearRegression

class Optimization(LinearRegression):
  ############## Train with stochastic Gradient Descent #####################
  def __init__(self) -> None:
    super().__init__()

  def train_sgd_descent(self,X, y, number_epoc):
    #number_epoc = int(input("Enter the number of Epoch :"))
    lr = 0.1#float(input("Enter the learning rate :"))
    n = X.shape[1]
    n_samples = X.shape[0]
    W = self.initialisation(n)
    store_loss = []
    epoch = 0
    loss_tolerance = 0.001
    avg_loss = float("inf")

    while epoch < number_epoc and avg_loss > loss_tolerance:
      running_loss = 0.0

      for e in range(X.shape[0]):
        sample_index = np.random.randint(0, n_samples-1)
        #print(lesample_index)
        samp_x = X[sample_index]#.reshape(-1, n)
        samp_y = y[sample_index]#.reshape(-1, 1)

        y_pred = self.make_prediction(samp_x,W)
        loss = self.mse(y_pred,samp_y)
        grad = self.gradient(samp_x,samp_y,W)
        W = self.update(W,grad,lr)
        print(f"\nEpoch {e}, loss {loss}")
        store_loss.append(loss)
      avg_loss = running_loss/ X.shape[0]
      return self.plot_loss(store_loss)

##################### Train with momementum ######################
  def get_momentum(self, momentum, grad, beta):
    return beta * momentum +(1-beta)*grad

  def train_sgd_momentum_descent(self, X, y, number_epoc, momentum, beta):
    lr=0.1
    n = X.shape[1]
    n_samples=X.shape[0]
    W = self.initialisation(n)
    store_loss = []
    for e in range(X.shape[0]):
      sample_index = np.random.randint(0, n_samples-1)
      #print(lesample_index)
      samp_x = X[sample_index]
      samp_y = y[sample_index]

      y_pred = self.make_prediction(samp_x,W)
      loss = self.mse(y_pred,samp_y)
      grad = self.gradient(samp_x,samp_y,W)
      momentum = self.get_momentum(momentum, grad, beta)
      W = self.update(W,momentum,lr)
      print(f"\nEpoch {e}, loss {loss}")
      store_loss.append(loss)
    return self.plot_loss(store_loss)

################### Train with mini bash ##########################

  def shuffle_data(self, X, y):
    N= X.shape[0]
    shuffled_idx = np.random.permutation(N)
    return X[shuffled_idx], y[shuffled_idx]

  def minibatch_gradient_descent(self, X, y, num_epochs, step_size=0.1, batch_size=3):
    n,d = X.shape
    theta = self.initialisation(d)
    losses = []
    num_batches = n//batch_size
    X, y = self.shuffle_data(X, y) # shuffle the data

    for epoch in range(num_epochs): # Do some iterations
      running_loss = 0.0

      for batch_idx in range(0, n, batch_size):
        x_batch = X[batch_idx: batch_idx + batch_size] # select a batch of features
        y_batch = y[batch_idx: batch_idx + batch_size] # and a batch of labels

        ypred = self.make_prediction(x_batch, theta) # make predictions with current parameters
        loss = self.mse(y_batch,ypred) # Compute mean squared error
        grads =  self.gradient(x_batch, y_batch, theta)# compute gradients of loss wrt parameters
        theta = self.update(theta, grads, step_size) # Update your parameters with the gradients
        running_loss += (loss * x_batch.shape[0]) # loss is mean for a batch, dividing by N_batch gives
                                                  # us a sum for the batch so we can average later by diving
                                                  # by the full data size

      avg_loss = running_loss/ n
      losses.append(avg_loss)
      print(f"\nEpoch {epoch}, loss {avg_loss}")

    return self.plot_loss(losses)