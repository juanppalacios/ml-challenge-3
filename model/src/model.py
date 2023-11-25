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

# np.random.seed(0)

# custom imports
from toolkit import Toolkit
from activation import ReLU, Sigmoid, Tanh, Softmax
from layer import Input, FullyConnected, Output
from loss import *

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

    self.loss_keys = {
      'mse'        : mean_square_error,
      'mse grad'   : mean_square_error_gradient,
      'cross entropy'      : cross_entropy_error,
      'cross entropy grad' : cross_entropy_error_gradient,
      'binary cross entropy'      : binary_cross_entropy_error,
      'binary cross entropy grad' : binary_cross_entropy_error_gradient,
    }

    self.loss = None
    self.loss_gradient = None

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

  def _set_loss(self, _loss):
    self.loss = self.loss_keys[_loss]
    self.loss_gradient = self.loss_keys[f"{_loss} grad"]

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

  def _train_model(self, _layers, _epochs, _samples, _learning_rate):

    _epoch_history = {'loss': [], 'accuracy': []}

    #> training loop
    for epoch in range(_epochs):
      _epoch_loss     = 0
      _epoch_accuracy = 0

      #> Shuffle the data and labels in the same order
      _shuffled_indices = np.random.permutation(_samples)
      _shuffled_data    = self.train_data[_shuffled_indices]
      _shuffled_labels  = self.train_labels[_shuffled_indices]

      for label, sample in zip(_shuffled_labels, _shuffled_data):
        label  = np.eye(10)[label]
        sample = sample.reshape((1, sample.shape[0]))

        #> forward propagation
        _output = sample
        for layer in _layers:
          _output = layer.forward(_output)

        #> finding epoch accuracy
        _epoch_accuracy += np.argmax(label) == np.argmax(_output)

        #> finding epoch loss and loss gradient
        _epoch_loss += self.loss(label, _output)
        _epoch_loss_gradient = self.loss_gradient(label, _output)

        #> backward propagation
        for layer in reversed(_layers):
          _epoch_loss_gradient = layer.backward(_epoch_loss_gradient, _learning_rate)

      _epoch_loss     /= _samples
      _epoch_accuracy = (100 * _epoch_accuracy) / _samples
      _epoch_history['loss'].append(_epoch_loss)
      _epoch_history['accuracy'].append(_epoch_accuracy)
      self.toolkit.info(f'epoch {epoch + 1}/{_epochs}, loss: {_epoch_loss:.1f}, accuracy: {_epoch_accuracy:.1f}%')

    return {
      'average loss'    : np.mean(_epoch_history['loss']),
      'average accuracy': np.mean(_epoch_history['accuracy'])
    }

  def fit(self, train_data = None, train_labels = None):

    self.train_data   = train_data
    self.train_labels = train_labels

    _samples = 1000
    # _samples = train_data.shape[0]
    _test_cases = self.parameters.length()
    self.layers = [[] for _ in range(_test_cases)]
    self.scores = [None for _ in range(_test_cases)]

    # create our neural architectures given current parameters
    for index, test_case in enumerate(self.parameters):

      # run a specific test case, skip the rest
      _selected_case = test_case['select_case']
      if _selected_case > -1:
        if index != _selected_case:
          continue

      # architecture parameters
      _hidden_layers     = test_case['hidden_layers']
      _hidden_dimensions = test_case['hidden_dimensions']
      _activations       = test_case['activation']

      # learning parameters
      _epochs        = test_case['epochs']
      _learning_rate = test_case['learning_rate']
      _loss          = test_case['loss']

      # set our loss function
      self._set_loss(_loss)

      # set our neural architecture
      self.layers[index] = self._set_architecture(_hidden_layers, _hidden_dimensions, _activations)

      self.toolkit.info(f"training case {index + 1}/{_test_cases}: {_epochs} epochs, {_learning_rate} learning rate ({_loss})")
      self.summary(index)

      # launch training loop
      self.scores[index] = self._train_model(self.layers[index], _epochs, _samples, _learning_rate)
      self.toolkit.info(f"test case summary: {self.scores[index]}")

    #> find the top test case with highest average accuracy
    if _selected_case == -1:
      self.index = max(range(len(self.scores)), key = lambda i : self.scores[i]['average accuracy'])
      self.toolkit.warning(f"highest index is {self.index}")
    else:
      self.index = _selected_case

  def predict(self, test_data, test_labels = None):

    self.test_data   = test_data
    self.test_labels = test_labels

    self.summary(self.index)

    for label, sample in zip(self.test_labels, self.test_data):
        sample = sample.reshape((1, sample.shape[0]))
        label  = label.reshape((1, label.shape[0]))

        #> forward propagation
        _output = sample
        for layer in self.layers[self.index]:
          _output = layer.forward(_output)

        #> append our answer
        label[0] = np.argmax(_output)

    return self.test_labels

  def evaluate(self, test_data, test_labels):
    raise NotImplementedError

  def summary(self, index):
    summary_str = '\nModel Summary:\n'
    case = self.layers[index]
    if len(case) == 0:
      self.toolkit.info(f'no layers defined!')
    else:
      for i, layer in enumerate(case):
        summary_str += f'\tlayer {i}: {layer}\n'
      self.toolkit.info(f'{summary_str}')
