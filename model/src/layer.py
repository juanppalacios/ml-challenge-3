from abc import ABC, abstractmethod
import numpy as np
from numba import njit, jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import logging;
logger = logging.getLogger("numba");
logger.setLevel(logging.ERROR)

# custom imports
from activation import BaseActivation, ReLU, Sigmoid, Tanh, Softmax


'''
______
| ___ \
| |_/ / __ _ ___  ___
| ___ \/ _` / __|/ _ \
| |_/ / (_| \__ \  __/
\____/ \__,_|___/\___|
'''

class BaseLayer():
  def __init__(self, input_size, output_size, activation):

    self.input_size  = input_size
    self.output_size = output_size
    self.activation  = activation

    self.input  = None
    self.output = None

    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias    = np.random.rand(1, output_size) - 0.5

  @abstractmethod
  def forward(self):
    raise NotImplementedError

  @abstractmethod
  def backward(self):
    raise NotImplementedError

  def shape(self):
    return (self.input_size, self.output_size)

  def __str__(self):
    return f"layer shape: ({self.shape()}), activation: {self.activation}"

'''
 _____                  _
|_   _|                | |
  | | _ __  _ __  _   _| |_
  | || '_ \| '_ \| | | | __|
 _| || | | | |_) | |_| | |_
 \___/_| |_| .__/ \__,_|\__|
           | |
           |_|
'''

class Input(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

  def forward(self, input):
    self.input  = input
    self.output = self.input
    return self.output

  def backward(self, output_error, learning_rate):
    pass
    # return output_error

  def __str__(self):
    return f'Input {super().__str__()}'

'''
______     _ _       _____                             _           _
|  ___|   | | |     /  __ \                           | |         | |
| |_ _   _| | |_   _| /  \/ ___  _ __  _ __   ___  ___| |_ ___  __| |
|  _| | | | | | | | | |    / _ \| '_ \| '_ \ / _ \/ __| __/ _ \/ _` |
| | | |_| | | | |_| | \__/\ (_) | | | | | | |  __/ (__| ||  __/ (_| |
\_|  \__,_|_|_|\__, |\____/\___/|_| |_|_| |_|\___|\___|\__\___|\__,_|
                __/ |
               |___/
'''

class FullyConnected(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

   # @jit(forceobj=True)
  def forward(self, input):
    self.input  = input
    self.output = self.activation.activate(np.dot(self.input, self.weights) + self.bias)
    return self.output

   # @jit(forceobj=True)
  def backward(self, output_gradient, learning_rate):
    # weight/bias contributiion to our error gradient
    weight_gradient = np.dot(self.input.T, output_gradient)
    # bias_gradient   = np.sum(output_gradient, axis = 0)

    # update weights/biases
    self.weights -= learning_rate * weight_gradient
    self.bias    -= learning_rate * output_gradient # bias_gradient

    return np.dot(output_gradient, self.weights.T) * self.activation.derivative(self.input)

  def __str__(self):
    return f'Fully Connected {super().__str__()}'

'''
 _____       _               _
|  _  |     | |             | |
| | | |_   _| |_ _ __  _   _| |_
| | | | | | | __| '_ \| | | | __|
\ \_/ / |_| | |_| |_) | |_| | |_
 \___/ \__,_|\__| .__/ \__,_|\__|
                | |
                |_|
'''

class Output(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

   # @jit(forceobj=True)
  def forward(self, input):
    self.input  = input
    self.output = self.activation.activate(np.dot(self.input, self.weights) + self.bias)
    return self.output

   # @jit(forceobj=True)
  def backward(self, output_gradient, learning_rate):
    # weight/bias contributiion to our error gradient
    weight_gradient = np.dot(self.input.T, output_gradient)
    bias_gradient   = np.sum(output_gradient, axis = 0)

    # update weights/biases
    self.weights -= learning_rate * weight_gradient
    self.bias    -= learning_rate * output_gradient # bias_gradient

    return np.dot(output_gradient, self.weights.T) * self.activation.derivative(self.input)

  def __str__(self):
    return f'Output {super().__str__()}'

'''
  References

  [1] Numpy Neural Network Github Repo
      https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py#L73-L137
'''

