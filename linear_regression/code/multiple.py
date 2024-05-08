## Multiple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])

  ## Your code here. Hint: You can use numpy to compute mean and std.
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)
  return mean, std


# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = np.zeros(X.shape)

  ## Your code here.
  S = (X - mean) / std
  return S

# Read data matrix X and labels t from text file.
def read_data(file_name):
  
  data=np.loadtxt(file_name)
  # Your code here. Load data features in X and labels in t.
  X = data[:, :-1]
  t = data[:, -1]
  return X, t


# Implement gradient descent algorithm to compute w = [w0, w1, ..].
def train(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])

  #  YOUR CODE here. Implement gradient descent to compute w for given epochs.
  #  Use 'compute_gradient' function below to find gradient of cost function and update w each epoch.
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.
  for epoch in range(epochs):
    grad = compute_gradient(X, t, w)
    w = w - eta * grad
    if epoch % 10 == 0:
      cost = compute_cost(X, t, w)
      costs.append(cost)
      ep.append(epoch)
  return w,ep,costs

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  cost = compute_cost(X, t, w)
  rmse = np.sqrt(2 * cost)
  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  N = len(t)
  h = np.dot(X, w)
  error = h - t
  cost = (1/(2*N)) * np.sum(error ** 2)
  return cost


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):
  grad = np.zeros(w.shape)
  # YOUR CODE here:
  N = len(t)
  h = np.dot(X, w)
  grad = (1/N) * np.dot(X.T, (h - t))

  return grad


# BONUS: Implement stochastic gradient descent algorithm to compute w = [w0, w1, ..].
def train_SGD(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  #  YOUR CODE here. Implement stochastic gradient descent to compute w for given epochs. 
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.
  # Iterate through epochs
  for epoch in range(epochs):
    # Iterate over each data point (Stochastic Gradient)
    for i in range(len(t)):
      x_i = X[i, :]  # Get features for current data point
      t_i = t[i]      # Get label for current data point
      h_i = np.dot(x_i, w)  # Predicted value for current data point
      grad_i = (1/len(t)) * (h_i - t_i) * x_i  # Gradient for current data point

      # Update weights using gradient
      w = w - eta * grad_i

    # Compute and store cost and epoch number every 10 epochs
    if epoch % 10 == 0:
      cost = compute_cost(X, t, w)
      costs.append(cost)
      ep.append(epoch)
  return w,ep,costs


##======================= Main program =======================##
# parser = argparse.ArgumentParser('Multiple Regression Exercise.')
# parser.add_argument('-i', '--input_data_dir',
#                     type=str,
#                     default='../data/multiple',
#                     help='Directory for the multiple regression houses dataset.')
# FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data("./data/multiple/train.txt")
Xtest, ttest = read_data("./data/multiple/test.txt")

#  YOUR CODE here: 
#  Standardize the training and test features using the mean and std computed over *training*.
#  Make sure you add the bias feature to each training and test example.
#  The bias features should be a column of ones addede as the first columns of training and test examples
mean, std = mean_std(Xtrain)
Xtrain = standardize(Xtrain, mean, std)
Xtrain = np.column_stack((np.ones(len(Xtrain)), Xtrain))

Xtest = standardize(Xtest, mean, std)
Xtest = np.column_stack((np.ones(len(Xtest)), Xtest))

# Computing parameters for each training method for eta=0.1 and 200 epochs
eta=0.1
epochs=200
w,eph,costs=train(Xtrain,ttrain,eta,epochs)
wsgd,ephsgd,costssgd=train_SGD(Xtrain,ttrain,eta,epochs)


# Print model parameters.
print('Params GD: ', w)
print('Params SGD: ', wsgd)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

# Print SGD results after training
trainRmseSgd = compute_rmse(Xtrain, ttrain, wsgd)
trainCostSgd = compute_cost(Xtrain, ttrain, wsgd)
testRmseSgd = compute_rmse(Xtest, ttest, wsgd)
testCostSgd = compute_cost(Xtest, ttest, wsgd)

print('Training RMSE SGD: %0.2f.' % trainRmseSgd)
print('Training cost SGD: %0.2f.' % trainCostSgd)
print('Test RMSE SGD: %0.2f.' % testRmseSgd)
print('Test cost SGD: %0.2f.' % testCostSgd)


# Plotting Epochs vs. cost for Gradient descent methods
plt.xlabel(' epochs')
plt.ylabel('cost')
plt.yscale('log')
plt.plot( eph,costs , 'bo-', label= 'train_Jw_gd')
plt.plot( ephsgd,costssgd , 'ro-', label= 'train_Jw_sgd')
plt.legend()
plt.savefig('gd_cost_multiple.png')
plt.close()