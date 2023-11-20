
import time
import numpy as np
from itertools import product

#> custom imports
from toolkit import Toolkit
from activation import ReLU, Sigmoid, Tanh, Softmax
from layer import Input, Flatten, FullyConnected, Output

class Model():
  def __init__(self, debug_mode = False):

    #> --> internal use
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
  def configure(self, parameters = None):
    '''
      set all the model's relevant training parameters like epoch, learning rate,
        and neural architecture
    '''

    if parameters is None:
      self.toolkit.error_out("model MUST include parameters!")

    self.parameters = parameters

    self.toolkit.debug(f"model received the following parameters: {self.parameters}")


    # note: we can create our layers now or during the fit step, we overwrite our layers

    # create our neural architectures given parameters
    # for index, test_case in enumerate(self.parameters):
      # # self.toolkit.info(f"case {index + 1}:  {type(test_case)} {test_case}")
      # for key, value in test_case.items():

      #   self.toolkit.info(f"{key} -> {value}")
      # note: for each element in our parameter list, create a unique layer list inside our






    # self.toolkit.info('model configured')

  # note: anything below this line is under construction

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

  def fit(self, train_data = None, train_labels = None):

    self.train_data   = train_data
    self.train_labels = train_labels

    # create our neural architectures given current parameters
    for index, test_case in enumerate(self.parameters):
      _epochs = test_case['epochs']
      _learning_rate = test_case['learning_rate']
      
      _samples = 2 # len(train_data)
      

      # todo: randomly initialize weights and biases

      # todo: create unique layer configuration

      self.toolkit.info(f'training our model for {_epochs} epochs with learning rate of {_learning_rate}')

      for epoch in range(_epochs):
        _epoch_error = 0.00
        for j in range(_samples):
          # todo: forward propagation
          # todo: error gradient
          # todo: backward propagation
          # todo: update weights and biases
          ...
        self.toolkit.debug(f'epoch {epoch + 1}/{_epochs}, error = {_epoch_error}')
        time.sleep(0.25)


    # todo: training loop


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