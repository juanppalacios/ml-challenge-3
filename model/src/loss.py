import numpy as np

class Loss:
  def __init__(self):
    ...

  # loss function and its derivative
  def error(self, y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

  def error_gradient(self, y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

