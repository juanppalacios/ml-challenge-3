import numpy as np
from tools import Toolkit, logging_levels
import time

'''
  Architecture inspired by Omar Aflak
  source: https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
'''

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

  def configure(self, debug_mode = False):
    if debug_mode:
      self.debug_mode = debug_mode
      self.toolkit.configure(name = 'MNIST Model', level = logging_levels['DEBUG'])
    else:
      self.toolkit.configure(name = 'MNIST Model', level = logging_levels['INFO'])

    self.toolkit.debug('running in debug mode!')

  def add_layer(self):
    self.layers.append()
    
  def summary(self):
    # todo: print out each layer: layer type, output shape, and parameter size
    ...

  def fit(self, x_train, y_train, epochs = 10, learning_rate = 0.01):
    self.x_train = x_train
    self.y_train = y_train

    samples = len(x_train)

    #> training loop
    self.toolkit.info(f'training our model for {epochs} epochs with learning rate of {learning_rate}')
    for i in range(epochs):
      error = 0.00
      for j in range(samples):
        # note: forward propagation
        pass
        # note: error gradient

        # note: backward propagation

        # note: update weights and biases
      self.toolkit.debug(f'epoch {i + 1}/{epochs}, error = {error}')
      time.sleep(1)

  def predict(self):
    pass

  def validate(self):
    pass
