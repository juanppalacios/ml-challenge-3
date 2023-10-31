import numpy as np
from tools import Toolkit, logging_levels

class Model():
  def __init__(self):
    self.debug_mode = False
    self.toolkit    = Toolkit()

  def configure(self, debug_mode = False):
    if debug_mode:
      self.debug_mode = debug_mode
      self.toolkit.configure(name = 'MNIST model logger', level = logging_levels['DEBUG'])
    else:
      self.toolkit.configure(name = 'MNIST model logger', level = logging_levels['INFO'])

    self.toolkit.debug('running in debug mode!')
