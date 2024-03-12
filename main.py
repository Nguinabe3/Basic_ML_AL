import numpy as np
import matplotlib.pyplot as plt 
from Linear_Reg import LinearRegression
from Logistic_Reg import LogisticRegression
from Optimi import Optimization

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
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

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

X = np.linspace(0,1, 10)
y = X + np.random.normal(0, 0.1, (10,))
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


LinReg = LinearRegression()
Optim = Optimization()
LogReg = LogisticRegression()

def main():
    print("Welcome to the Gradient Descent Algorithm Implementation Program")
    print("Please select which specific algorithm you would like to use:")
    print("1. Simple Basch Gradient Descent 1")
    print("2. Stochastic Gradient Descent 2")
    print("3. Stochastic Gradient Descent with Momentum 3")
    print("4. Min-Basch Gradient Descent 4")
    print("5. Logistic Regression 5")

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
      LinReg.train_basch_descent(X, y, int(input("Enter the number of epoch:")))
    elif choice == '2':
      Optim.train_sgd_descent(X, y, int(input("Enter the number of epoch:")))
    elif choice == '3':
      Optim.train_sgd_momentum_descent(X, y, int(input("Enter the number of epoch:")) ,float(input("Enter the value of beta:")), beta=float(input("Enter the value of beta:")))
    elif choice == '4':
      Optim.minibatch_gradient_descent(X, y, int(input("Enter the number of epoch:")), step_size = float(input("Enter the your stepsize:")), batch_size=int(input("Enter the batch_size:")))
    elif choice == '5':
      LogReg.fit(X_train, y_train, float(input("Enter the learning rate")), int(input("Enter the number of epoch ")))
      ypred_train=LogReg.predict(X_train)
      acc = LogReg.accuracy(y_train,ypred_train)
      print(f"The training accuracy is: {acc}")
    else:
        print("Invalid choice. Please select a valid option.")
if __name__ == "__main__":
    main()