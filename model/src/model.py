from itertools import product
import numpy as np
from toolkit import Toolkit, logging_levels
import time

class Layers():
  pass

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
    else:
      self.toolkit.configure(name = 'MNIST Model', level = logging_levels['INFO'])

    self.toolkit.debug('running in debug mode!')

    self.parameters = parameters

    self.scores     = [0.0  for _ in range(len(self.parameters))]
    self.y_test     = [None for _ in range(len(self.parameters))]

    self.toolkit.debug(f'we will store a total of {len(self.scores)} scores and y_tests')

  def add_layer(self):
    self.layers.append(None)

  def summary(self):
    # todo: print out each layer: layer type, output shape, and parameter size
    pass

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
    self.toolkit.info('predicting output')

  def validate(self):
    self.toolkit.info('validating model with binary cross-entropy')

  def evaluate(self, x_test, y_test):
    self.toolkit.info('validating model with binary cross-entropy')
    return self.scores

'''
  References

  [1] Neural Networl Architecture inspired by Omar Aflak
      https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
  [2] Tensorflow 2 Documentation
      https://www.tensorflow.org/guide/keras/functional_api
  [3] ...
'''