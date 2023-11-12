
from itertools import product
import numpy as np
import time

from toolkit import Toolkit, logging_levels

from activation import ReLU, Sigmoid, Tanh, Softmax
from layer import Input, Flatten, FullyConnected, Output

activation_keys = {
  'none'    : None,
  'relu'    : ReLU(),
  'sigmoid' : Sigmoid(),
  'tanh'    : Tanh(),
  'softmax' : Softmax()
}


class Model():
  def __init__(self):
    self.debug_mode = False
    self.toolkit    = Toolkit()

    self.x_train  = None
    self.y_train  = None
    self.x_test   = None
    self.y_test   = None
    self.y_golden = None

    self.layers = []

    self.error  = 0.0

    self.parameters = None
    self.scores     = {'accuracy' : 0.00, 'parameters' : [None]}
    self.y_pred     = None

  def configure(self, parameters = None, debug_mode = False):

    if debug_mode:
      self.debug_mode = debug_mode
      self.toolkit.configure(name = 'MNIST Model', level = logging_levels['DEBUG'])
      self.toolkit.debug('running in debug mode!')
    else:
      self.toolkit.configure(name = 'MNIST Model', level = logging_levels['INFO'])

    self.parameters = parameters
    self.scores     = [0.0  for _ in range(len(self.parameters))]
    self.y_test     = [None for _ in range(len(self.parameters))]

    for parameter in self.parameters:
      self.toolkit.info(f'{parameter}')

  def set_architecture(self, parameter):
    self.toolkit.debug(f'according to these parameters: ')

  def add_input_layer(self, input_size, output_size, activation = None):
    self.layers.append(Input(input_size, output_size, activation_keys[activation]))
    self.toolkit.debug(f'added {self.layers[-1]}')

  def add_hidden_layer(self, input_size, output_size, activation = None):
    self.layers.append(FullyConnected(input_size, output_size, activation_keys[activation]))
    if self.layers[-2].dimensions()[1] != self.layers[-1].dimensions()[0]:
      self.toolkit.error_out(f'layer dimension mismatch: {self.layers[-2].dimensions()[1]} != {self.layers[-1].dimensions()[0]}')
    self.toolkit.debug(f'added {self.layers[-1]}')

  def add_output_layer(self, input_size, output_size, activation = None):
    self.layers.append(Output(input_size, output_size, activation_keys[activation]))
    if self.layers[-2].dimensions()[1] != self.layers[-1].dimensions()[0]:
      self.toolkit.error_out(f'layer dimension mismatch: {self.layers[-2].dimensions()[1]} != {self.layers[-1].dimensions()[0]}')
    self.toolkit.debug(f'added {self.layers[-1]}')

  def fit(self, x_train, y_train, epochs = 10, learning_rate = 0.01):
    self.x_train = x_train
    self.y_train = y_train

    samples = len(x_train)

    #> training loop
    self.toolkit.info(f'training our model for {epochs} epochs with learning rate of {learning_rate}')

    # note: randomly initialize weights and biases
    for i in range(epochs):
      error = 0.00
      for j in range(samples):
        # note: forward propagation
        pass
        # note: error gradient

        # note: backward propagation

        # note: update weights and biases
      self.toolkit.debug(f'epoch {i + 1}/{epochs}, error = {error}')
      time.sleep(0.25)

  def predict(self, x_test, y_golden = None):
    raise NotImplementedError

  def evaluate(self, x_test, y_test):
    raise NotImplementedError

  def summary(self):
    summary_str = 'Model Summary:\n'
    for index, layer in enumerate(self.layers):
      summary_str += f'\tlayer {index}: {layer}\n'
    self.toolkit.info(f'{summary_str}')

'''
  References

  [1] Neural Networl Architecture inspired by Omar Aflak
      https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
  [2] Tensorflow 2 Documentation
      https://www.tensorflow.org/guide/keras/functional_api
  [3] ...
'''