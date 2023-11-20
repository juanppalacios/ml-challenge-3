
import time
import numpy as np
from itertools import product

#> custom imports
from toolkit import Toolkit
from activation import ReLU, Sigmoid, Tanh, Softmax
from layer import Input, Flatten, FullyConnected, Output


class Model():
  def __init__(self):
    
    #> --> internal use
    self.debug_mode = False
    self.toolkit    = Toolkit()

    self.activation_keys = {
      'none'    : None,
      'relu'    : ReLU(),
      'sigmoid' : Sigmoid(),
      'tanh'    : Tanh(),
      'softmax' : Softmax()
    }

    #> --> model.configure()
    self.layers     = [] # our layers may need to change?
    # self.layers = {}
    self.parameters = None

    #> --> model.fit()
    self.train_data   = None # single np array with all data
    self.train_labels = None # single np array with all labels
    self.test_data    = None # single np array with all data
    self.test_labels  = []   # multiple np arrays with all labels --> returns the best one
    self.test_golden  = None # single np array with all labels --> used for verification

    #> --> model.predict/evaluate()
    self.scores = [] # {'accuracy' : 0.00, 'parameters' : [None]}
    # self.y_pred     = None
    # self.error      = 0.0

  # note: finish this first before fitting training data
  def configure(self, parameters = None, debug_mode = False):

    #> when running in `debug_mode`, spawn a debugger logger, otherwise, keep the model's default logger
    self.debug_mode = debug_mode
    if self.debug_mode:
      self.toolkit.configure(name = 'MNIST Model Debugger', level = 'DEBUG')
      self.toolkit.debug('running in debug mode!')
    else:
      self.toolkit.configure(name = 'MNIST Model', level = 'INFO')

    if parameters is None:
      self.toolkit.error_out("configuring a model MUST include parameters!")

    self.parameters  = parameters

    # self.test_labels = [np.zeros(1) for _ in range(len(self.parameters))]
    # self.scores     = [None for _ in range(len(self.parameters))]
    # self.y_pred     = [None for _ in range(len(self.parameters))]

    #> creating our custom architecture, note: we are HARDCODING for now, may need to switch to KWARGS
    for case in parameters:
      # self.toolkit.info(f"test case: {case}")
      for parameter in case:
        if isinstance(parameter, list):
          self.toolkit.info(f"test case: {parameter}")
          
          # self.layers.append()
          
      break

    self.toolkit.info('model configured')

  def add_input_layer(self, input_size, output_size, activation = None):
    self.layers.append(Input(input_size, output_size, self.activation_keys[activation]))
    self.toolkit.debug(f'added {self.layers[-1]}')

  def add_hidden_layer(self, input_size, output_size, activation = None):
    self.layers.append(FullyConnected(input_size, output_size, self.activation_keys[activation]))
    if self.layers[-2].dimensions()[1] != self.layers[-1].dimensions()[0]:
      self.toolkit.error_out(f'layer dimension mismatch: {self.layers[-2].dimensions()[1]} != {self.layers[-1].dimensions()[0]}')
    self.toolkit.debug(f'added {self.layers[-1]}')

  def add_output_layer(self, input_size, output_size, activation = None):
    self.layers.append(Output(input_size, output_size, self.activation_keys[activation]))
    if self.layers[-2].dimensions()[1] != self.layers[-1].dimensions()[0]:
      self.toolkit.error_out(f'layer dimension mismatch: {self.layers[-2].dimensions()[1]} != {self.layers[-1].dimensions()[0]}')
    self.toolkit.debug(f'added {self.layers[-1]}')





  # note: under construction
  def fit(self, train_data, train_labels, epochs = 10, learning_rate = 0.01):
    self.train_data = train_data
    self.train_labels = train_labels

    samples = len(train_data)

    # todo: training loop
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

  def predict(self, test_data, test_golden = None):
    raise NotImplementedError

  def evaluate(self, test_data, test_labels):
    raise NotImplementedError

  def summary(self):
    summary_str = 'Model Summary:\n'
    if len(self.layers) == 0:
      self.toolkit.info(f'no layers defined!')
    else:
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

  Slowly integrate calls back in...
  # mnist_model.configure(
  #   parameters   = parameters.all(),
  #   debug_mode   = False
  # )

  # mnist_model.add_input_layer(784, 64, 'none')
  # mnist_model.add_hidden_layer(64, 32, 'relu')
  # mnist_model.add_hidden_layer(32, 16, 'sigmoid')
  # mnist_model.add_hidden_layer(16, 8, 'tanh')
  # mnist_model.add_output_layer(8, 10, 'softmax')

  # mnist_model.summary()

  # mnist_model.fit(train_data, train_labels, epochs = 10, learning_rate = 0.01)

  # scores = mnist_model.evaluate(test_data, test_labels)

  #> reporting our highest score with parameters
  # tools.info(f'high score of {scores['accuracy']} ran with parameters {scores['parameters']}')
'''