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
    for case in self.test_cases:
      args += f'\t{case}\n'
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

  def load_data(self, path):
    pass

  def save_data(self, path, data):
    pass



import itertools

# todo: find a way to integrate this to our code
# note: this code generates our parameter combinations
if False:
# Step 1: Define layer configurations
  num_layers = [2, 3]
  activation_functions_hidden = ['relu', 'sigmoid']  # Define possible activation functions for hidden layers

  # Define possible total node values for hidden layers as a list
  hidden_layer_nodes = [64, 32, 16, 8, 4]

  input_nodes = 28*28  # Fixed input layer with 28 nodes
  output_nodes = 10  # Fixed output layer with 10 nodes

  # Step 2: Use itertools.product to generate combinations
  encoded_combinations = []
  for num_hidden_layers in num_layers:
      activation_combinations = list(itertools.product(activation_functions_hidden, repeat=num_hidden_layers))

      for activation_combo in activation_combinations:
          # Create a list of (input, output) node tuples for hidden layers, including input and output layers
          hidden_layer_tuples = [(input_nodes, hidden_layer_nodes[0])]
          for i in range(num_hidden_layers):
              final_node = hidden_layer_nodes[i + 1] if i < num_hidden_layers - 1 else output_nodes
              hidden_layer_tuples.append((hidden_layer_nodes[i], final_node))

          encoded_combination = {
              'num_hidden_layers': num_hidden_layers,
              'input_output_nodes_hidden': hidden_layer_tuples,
              # 'input_nodes': input_nodes,
              # 'output_nodes': output_nodes,
              'activation_functions': activation_combo
          }
          encoded_combinations.append(encoded_combination)

  # Now, encoded_combinations contains all possible layer configurations with variable numbers of hidden layers
  for combination in encoded_combinations:
      print(combination)
