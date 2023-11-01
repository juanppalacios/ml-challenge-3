import numpy   as np
import logging as log
from itertools import product

logging_levels = {
  'DEBUG'  : log.DEBUG,
  'INFO'   : log.INFO,
  'WARNING': log.WARNING,
  'ERROR'  : log.ERROR,
}

class ParameterManager:
  def __init__(self):
    self.parameters = {}
    self.test_cases = []

  def add_parameter(self, arg, values):
    self.parameters[arg] = values
    self.test_cases = list(product(*self.parameters.values()))

  def get_parameter(self, arg):
    # note: removal tag
    return self.parameters[arg] if arg in self.parameters else None

  def all(self):
    return self.test_cases

  def __repr__(self):
    args = '\n'
    for key, value in self.parameters.items():
      args += f'\t{key} -> {value}\n'
    return args

class Toolkit():
  def __init__(self):
    #> logging utilities
    self.logger    = None
    self.formatter = None
    self.handler   = None

    #> file i/o utilities

  def configure(self, name, level = log.DEBUG):

    self.logger    = log.getLogger(f'{name}')
    self.formatter = log.Formatter('%(name)s - %(levelname)s - %(message)s')
    self.handler   = log.StreamHandler()

    self.handler.setFormatter(self.formatter)
    self.logger.addHandler(self.handler)
    self.logger.setLevel(level)

  '''
  LOGGING
  '''

  def debug(self, message):
    self.logger.debug(message)

  def info(self, message):
    self.logger.info(message)

  def warning(self, message):
    self.logger.warning(message)

  def error_out(self, message):
    self.logger.error(message)
    exit(1)
    
  '''
  
  '''

  '''
  DATA PROCESSOR
  '''
  def normalize(self, data):
    return data

  '''
  FILE I/O
  '''

  def read_input(self, path):
    pass

  def write_output(self, path, data):
    pass