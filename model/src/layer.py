from abc import ABC, abstractmethod
import numpy as np

# custom imports
from activation import BaseActivation, ReLU, Sigmoid, Tanh, Softmax
from optimizer import AdamOptimizer

np.random.seed(0)

'''
______
| ___ \
| |_/ / __ _ ___  ___
| ___ \/ _` / __|/ _ \
| |_/ / (_| \__ \  __/
\____/ \__,_|___/\___|
'''

class BaseLayer():
  def __init__(self, input_size, output_size, activation):

    self.input_size  = input_size
    self.output_size = output_size
    self.activation  = activation

    self.input  = None
    self.output = None

  @abstractmethod
  def forward(self):
    raise NotImplementedError

  @abstractmethod
  def backward(self):
    raise NotImplementedError

  def shape(self):
    return (self.input_size, self.output_size)

  def __str__(self):
    return f"layer shape: ({self.shape()}), activation: {self.activation}"

'''
 _____                  _
|_   _|                | |
  | | _ __  _ __  _   _| |_
  | || '_ \| '_ \| | | | __|
 _| || | | | |_) | |_| | |_
 \___/_| |_| .__/ \__,_|\__|
           | |
           |_|
'''

class Input(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

  def forward(self, input):
    self.input  = input
    self.output = self.input
    return self.output

  def backward(self, output_error, learning_rate):
    return output_error
    pass

  def __str__(self):
    return f'Input {super().__str__()}'

'''
______     _ _       _____                             _           _
|  ___|   | | |     /  __ \                           | |         | |
| |_ _   _| | |_   _| /  \/ ___  _ __  _ __   ___  ___| |_ ___  __| |
|  _| | | | | | | | | |    / _ \| '_ \| '_ \ / _ \/ __| __/ _ \/ _` |
| | | |_| | | | |_| | \__/\ (_) | | | | | | |  __/ (__| ||  __/ (_| |
\_|  \__,_|_|_|\__, |\____/\___/|_| |_|_| |_|\___|\___|\__\___|\__,_|
                __/ |
               |___/
'''

class FullyConnected(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

    # Create an instance of Adam optimizer
    self.optimizer = AdamOptimizer(beta1=0.9, beta2=0.999, epsilon=1e-8)
    self.timestep = 0

    self.weights = np.random.rand(input_size, output_size) * 0.01
    self.bias    = np.random.rand(1, output_size) * 0.01

  def forward(self, input):
    self.input  = input

    # apply lienar transformation
    _linear     = np.dot(self.input, self.weights) + self.bias

    # apply non-linear activation
    self.output = self.activation.activate(_linear)

    return self.output

  def backward(self, output_gradient, learning_rate):
    input_gradient  = np.dot(output_gradient, self.weights.T)
    weight_gradient = np.dot(self.input.T, output_gradient)
    bias_gradient   = np.mean(output_gradient, axis = 1, keepdims = True)

    # increment time step for adam optimizer
    self.timestep += 1

    # Update parameters using Adam optimizer
    gradients = {'weights': weight_gradient, 'bias': bias_gradient}
    parameter_updates = self.optimizer.update(learning_rate, gradients, self.timestep)

    # update weights/biases
    self.weights -= learning_rate * weight_gradient + parameter_updates['weights']
    self.bias    -= learning_rate * bias_gradient + parameter_updates['bias']

    # return input_gradient
    return self.activation.derivative(self.input) * input_gradient

    # # weight/bias contribution wrt to loss
    # weight_gradient = np.dot(self.input.T, output_gradient)
    # bias_gradient   = np.mean(output_gradient, axis = 1, keepdims = True)

    # # find input gradient
    # input_gradient = np.dot(output_gradient, self.weights.T) * self.activation.derivative(self.input)

    # # update weights/biases
    # self.weights -= learning_rate * weight_gradient
    # self.bias    -= learning_rate * bias_gradient

    return input_gradient

  def __str__(self):
    return f'Fully Connected {super().__str__()}'

'''
 _____       _               _
|  _  |     | |             | |
| | | |_   _| |_ _ __  _   _| |_
| | | | | | | __| '_ \| | | | __|
\ \_/ / |_| | |_| |_) | |_| | |_
 \___/ \__,_|\__| .__/ \__,_|\__|
                | |
                |_|
'''

class Output(BaseLayer):
  def __init__(self, input_size, output_size, activation):
    super().__init__(input_size, output_size, activation)

    # Create an instance of Adam optimizer
    self.optimizer = AdamOptimizer(beta1=0.9, beta2=0.999, epsilon=1e-8)
    self.timestep = 0

    self.weights = np.random.rand(input_size, output_size) * 0.01
    self.bias    = np.random.rand(1, output_size) * 0.01

  def forward(self, input):
    self.input  = input

    # apply lienar transformation
    _linear     = np.dot(self.input, self.weights) + self.bias

    # apply non-linear activation
    self.output = self.activation.activate(_linear)

    return self.output

  def backward(self, output_gradient, learning_rate):
    input_gradient  = np.dot(output_gradient, self.weights.T)
    weight_gradient = np.dot(self.input.T, output_gradient)
    bias_gradient   = np.mean(output_gradient, axis = 1, keepdims = True)

    # increment time step for adam optimizer
    self.timestep += 1

    # Update parameters using Adam optimizer
    gradients = {'weights': weight_gradient, 'bias': bias_gradient}
    parameter_updates = self.optimizer.update(learning_rate, gradients, self.timestep)

    # update weights/biases
    self.weights -= learning_rate * weight_gradient + parameter_updates['weights']
    self.bias    -= learning_rate * bias_gradient + parameter_updates['bias']

    # return input_gradient
    return self.activation.derivative(self.input) * input_gradient
    # # weight/bias contribution wrt to loss
    # weight_gradient = np.dot(self.input.T, output_gradient)
    # bias_gradient   = np.mean(output_gradient, axis = 1, keepdims = True)

    # # find input gradient
    # input_gradient = np.dot(output_gradient, self.weights.T) * self.activation.derivative(self.input)

    # # update weights/biases
    # self.weights -= learning_rate * weight_gradient
    # self.bias    -= learning_rate * bias_gradient

    return input_gradient

  def __str__(self):
    return f'Output {super().__str__()}'

'''
  References

  [1] Numpy Neural Network Github Repo
      https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py#L73-L137
'''

