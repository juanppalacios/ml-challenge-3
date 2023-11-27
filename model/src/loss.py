import numpy as np

# class Loss:
  # def __init__(self):
  #   ...

  # def mean_square_error(self, target, predicted):
  #   return np.mean(np.power(target - predicted, 2))

  # def mean_square_error_gradient(self, target, predicted):
  #   return 2 * (predicted - target) / np.size(target)

  # def cross_entropy_error(self, target, predicted, epsilon = 1e-15):
  #   predicted = np.clip(predicted, epsilon, 1 - epsilon)
  #   losses = -np.sum(target * np.log(predicted), axis = 1)
  #   return np.mean(losses)

  # def cross_entropy_error_gradient(self, target, predicted, epsilon = 1e-15):
  #   predicted = np.clip(predicted, epsilon, 1 - epsilon)
  #   gradient = -target / predicted
  #   return gradient / len(target)

  # def binary_cross_entropy(target, predicted):
  #     return np.mean(-target * np.log(predicted) - (1 - target) * np.log(1 - predicted))

  # def binary_cross_entropy_prime(target, predicted):
  #     return ((1 - target) / (1 - predicted) - target / predicted) / np.size(target)

def mean_square_error(target, predicted):
  return np.mean(np.power(target - predicted, 2))

def mean_square_error_gradient(target, predicted):
  return 2 * (predicted - target)

def cross_entropy_error(target, predicted, epsilon=1e-10):
  predicted = np.clip(predicted, epsilon, 1 - epsilon)
  losses = -np.sum(target * np.log(predicted + epsilon), axis=1)
  return np.mean(losses)

def cross_entropy_error_gradient(target, predicted, epsilon=1e-10):
  return -(target / (np.clip(predicted, epsilon, 1 - epsilon) + epsilon))

# def cross_entropy_error(target, predicted, epsilon = 1e-10):
#   predicted = np.clip(predicted, epsilon, 1 - epsilon)
#   losses = -np.sum(target * np.log(predicted), axis = 1)
#   return np.mean(losses)

# def cross_entropy_error_gradient(target, predicted, epsilon = 1e-10):
#   return -target / np.clip(predicted, epsilon, 1 - epsilon)

def binary_cross_entropy_error(target, predicted):
  return np.mean(-target * np.log(predicted) - (1 - target) * np.log(1 - predicted))

def binary_cross_entropy_error_gradient(target, predicted):
  return ((1 - target) / (1 - predicted) - target / predicted) / np.size(target)