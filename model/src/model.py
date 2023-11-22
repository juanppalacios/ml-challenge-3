'''
  References

  [1] Neural Networl Architecture inspired by Omar Aflak
      https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
  [2] Tensorflow 2 Documentation
      https://www.tensorflow.org/guide/keras/functional_api
  [3] ...
'''

import time
from itertools import product
import numpy as np

from numba import jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from rich.progress import track

# custom imports
from toolkit import Toolkit
from activation import ReLU, Sigmoid, Tanh, Softmax
from layer import Input, FullyConnected, Output
from loss import Loss

class Model():
  def __init__(self, debug_mode = False):

    self.debug_mode = debug_mode
    self.toolkit    = Toolkit()

    # when running in `debug_mode`, spawn a debugger logger, otherwise, keep the model's default logger
    if self.debug_mode:
      self.toolkit.configure(name = 'MNIST Model Debugger', level = 'DEBUG')
      self.toolkit.debug('running in debug mode!')
    else:
      self.toolkit.configure(name = 'MNIST Model', level = 'INFO')

    self.activation_keys = {
      'none'    : None,
      'relu'    : ReLU(),
      'sigmoid' : Sigmoid(),
      'tanh'    : Tanh(),
      'softmax' : Softmax()
    }

    self.loss = Loss()

    #> --> model.configure()
    self.layers     = []
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

  def configure(self, parameters = None):
    '''
      set all the model's relevant training parameters like epoch, learning rate,
        and neural architecture
    '''

    if parameters is None:
      self.toolkit.error_out("model MUST include parameters!")

    self.parameters = parameters

    self.toolkit.info('model configured')

  # note: anything below this line is under construction

  def _set_architecture(self, _hidden_layers, _hidden_dimensions, _activations):
    _neural_architecture = [] # clear this unique model's layer configuration

    # append our input layer
    _neural_architecture.append(
      Input(_hidden_dimensions[0][0], _hidden_dimensions[0][0], 'none')
    )

    # append our hidden layers
    for new_layer in range(_hidden_layers):
      _neural_architecture.append(
        FullyConnected(_hidden_dimensions[new_layer][0], _hidden_dimensions[new_layer][1], self.activation_keys[_activations[new_layer]])
      )

    # append our output layer
    _neural_architecture.append(
      Output(_hidden_dimensions[_hidden_layers][0], _hidden_dimensions[_hidden_layers][1], self.activation_keys[_activations[_hidden_layers]])
    )

    return _neural_architecture

  def _train_model(self):
    raise NotImplementedError

  def fit(self, train_data = None, train_labels = None):

    self.train_data   = train_data
    self.train_labels = train_labels

    _samples = 1000 # train_data.shape[0]
    _test_cases = self.parameters.length()

    # create our neural architectures given current parameters
    for index, test_case in enumerate(self.parameters):

      # architecture parameters
      _hidden_layers = test_case['hidden_layers']
      _hidden_dimensions = test_case['hidden_dimensions']
      _activations = test_case['activation']

      # learning parameters
      _epochs = test_case['epochs']
      _learning_rate = test_case['learning_rate']

      # set our neural architecture
      self.layers = self._set_architecture(_hidden_layers, _hidden_dimensions, _activations)

      self.toolkit.info(f"training test case {index + 1} of {_test_cases} for {_epochs} epochs, learning rate is {_learning_rate} ")
      self.summary()

      #> training loop
      # for epoch in track(range(_epochs), description = f'test case {index + 1}...'):
      for epoch in range(_epochs):
        _epoch_error = 0
        # _epoch_error_gradient = 0
        # for sample in track(range(_samples), description = f'train epoch {epoch + 1}...'):
        for sample in range(_samples):
          #> forward propagation
          _output = self.train_data[sample, :].reshape((1, self.train_data.shape[1]))
          for layer in self.layers:
            # self.toolkit.debug(f"forward current layer: {layer} input: {type(_output)}")
            _output = layer.forward(_output)

          #> error gradient
          _epoch_error += self.loss.error(self.train_labels[sample], _output)

          #> backward propagation
          _epoch_error_gradient = self.loss.error_gradient(self.train_labels[sample], _output)
          for layer in reversed(self.layers):
            # self.toolkit.debug(f"backward current layer: {layer},\nerror gradient: {type(_epoch_error_gradient.shape)}")
            _epoch_error_gradient = layer.backward(_epoch_error_gradient, _learning_rate)

        _epoch_error /= _samples
        self.toolkit.info(f'epoch {epoch + 1}/{_epochs}, error = {_epoch_error}')

  def predict(self, test_data, test_labels):
    raise NotImplementedError

  def evaluate(self, test_data, test_labels):
    raise NotImplementedError

  def summary(self):
    summary_str = '\nModel Summary:\n'
    if len(self.layers) == 0:
      self.toolkit.info(f'no layers defined!')
    else:
      for index, layer in enumerate(self.layers):
        summary_str += f'\tlayer {index}: {layer}\n'
      self.toolkit.info(f'{summary_str}')
