from abc import ABC, abstractmethod
from activation import BaseActivation, ReLU, Sigmoid, Tanh, Softmax
import numpy as np

# todo: flush out weights/biases

'''
______
| ___ \
| |_/ / __ _ ___  ___
| ___ \/ _` / __|/ _ \
| |_/ / (_| \__ \  __/
\____/ \__,_|___/\___|
'''

class BaseLayer():
  def __init__(self, input_size, output_size, activation_function):
    self.input_size = input_size
    self.output_size = output_size
    self.activation_function = activation_function

  @abstractmethod
  def forward(self):
    raise NotImplementedError

  @abstractmethod
  def backward(self):
    raise NotImplementedError

  def dimensions(self):
    return (self.input_size, self.output_size)

  def __str__(self):
    return f"Layer: ({self.input_size} -> {self.output_size}), activation: {self.activation_function}"

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
  def __init__(self, input_size, output_size, activation_function):
    super().__init__(input_size, output_size, activation_function)

  def __str__(self):
    return f'Input {super().__str__()}'

'''
______ _       _   _
|  ___| |     | | | |
| |_  | | __ _| |_| |_ ___ _ __
|  _| | |/ _` | __| __/ _ \ '_ \
| |   | | (_| | |_| ||  __/ | | |
\_|   |_|\__,_|\__|\__\___|_| |_|
'''

class Flatten(BaseLayer):
  def __init__(self, input_size, output_size, activation_function):
    super().__init__(input_size, output_size, activation_function)

    self.weights = np.zeros((input_size, output_size))
    self.bias    = np.zeros(output_size)

  def __str__(self):
    return f'Flatten {super().__str__()}'

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
  def __init__(self, input_size, output_size, activation_function):
    super().__init__(input_size, output_size, activation_function)

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
  def __init__(self, input_size, output_size, activation_function):
    super().__init__(input_size, output_size, activation_function)

  def __str__(self):
    return f'Output {super().__str__()}'

'''
  References

  [1] Numpy Neural Network Github Repo
      https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py#L73-L137
'''

