import numpy as np
from tools import Toolkit, logging_levels

class Model():
  def __init__(self):
    self.debug_mode = False
    self.toolkit    = Toolkit()

    self.x_train  = None
    self.y_train  = None
    self.x_test   = None
    self.y_test   = None
    self.y_golden = None

  def configure(self, debug_mode = False):
    if debug_mode:
      self.debug_mode = debug_mode
      self.toolkit.configure(name = 'MNIST model', level = logging_levels['DEBUG'])
    else:
      self.toolkit.configure(name = 'MNIST model', level = logging_levels['INFO'])

    self.toolkit.debug('running in debug mode!')

  def fit(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train
