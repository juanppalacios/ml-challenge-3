from abc import ABC, abstractmethod
import numpy as np
from numba import jit, jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import logging;
logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)


'''
______
| ___ \
| |_/ / __ _ ___  ___
| ___ \/ _` / __|/ _ \
| |_/ / (_| \__ \  __/
\____/ \__,_|___/\___|
'''

class BaseActivation():
  def __init__(self) -> None:
    pass

  @abstractmethod
  def activate(self):
    raise NotImplementedError

  @abstractmethod
  def derivative(self):
    raise NotImplementedError

'''
______     _     _   _
| ___ \   | |   | | | |
| |_/ /___| |   | | | |
|    // _ \ |   | | | |
| |\ \  __/ |___| |_| |
\_| \_\___\_____/\___/
'''

class ReLU(BaseActivation):
  def __init__(self) -> None:
    super().__init__()

   # @jit(forceobj=True)
  def activate(self, x : np.ndarray):
    return np.maximum(x, 0)

   # @jit(forceobj=True)
  def derivative(self, x : np.ndarray):
    return np.where(x > 0, 1, 0)

  def __repr__(self) -> str:
    return "ReLU"

'''
 _____ _                       _     _
/  ___(_)                     (_)   | |
\ `--. _  __ _ _ __ ___   ___  _  __| |
 `--. \ |/ _` | '_ ` _ \ / _ \| |/ _` |
/\__/ / | (_| | | | | | | (_) | | (_| |
\____/|_|\__, |_| |_| |_|\___/|_|\__,_|
          __/ |
         |___/
'''

class Sigmoid(BaseActivation):
  def __init__(self) -> None:
    super().__init__()

   # @jit(forceobj=True)
  def activate(self, x : np.ndarray):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

   # @jit(forceobj=True)
  def derivative(self, x : np.ndarray):
    return self.activate(x) * (1 - self.activate(x))

  def __repr__(self) -> str:
    return "Sigmoid"

'''
 _____           _
|_   _|         | |
  | | __ _ _ __ | |__
  | |/ _` | '_ \| '_ \
  | | (_| | | | | | | |
  \_/\__,_|_| |_|_| |_|
'''

class Tanh(BaseActivation):
  def __init__(self) -> None:
    super().__init__()

   # @jit(forceobj=True)
  def activate(self, x):
    return np.tanh(x)

   # @jit(forceobj=True)
  def derivative(self, x):
    return 1 - np.tanh(x) ** 2

  def __repr__(self) -> str:
    return "Tanh"

'''
 _____        __ _         _
/  ___|      / _| |       | |
\ `--.  ___ | |_| |_ _ __ | |_   _ ___
 `--. \/ _ \|  _| __| '_ \| | | | / __|
/\__/ / (_) | | | |_| |_) | | |_| \__ \
\____/ \___/|_|  \__| .__/|_|\__,_|___/
                    | |
                    |_|
'''

class Softplus(BaseActivation):
  def __init__(self) -> None:
    super().__init__()

   # @jit(forceobj=True)
  def activate(self, x):
    return np.log(1 + np.exp(x))

   # @jit(forceobj=True)
  def derivative(self, x):
    return 1 / (1 + np.exp(-x))

  def __repr__(self) -> str:
    return "Softplus"

'''
 _____        __ _
/  ___|      / _| |
\ `--.  ___ | |_| |_ _ __ ___   __ ___  __
 `--. \/ _ \|  _| __| '_ ` _ \ / _` \ \/ /
/\__/ / (_) | | | |_| | | | | | (_| |>  <
\____/ \___/|_|  \__|_| |_| |_|\__,_/_/\_\
'''

class Softmax(BaseActivation):
  def __init__(self) -> None:
    super().__init__()

   # @jit(forceobj=True)
  def activate(self, x : np.ndarray):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    # e_x = np.exp(x - np.max(x))
    # return e_x / np.sum(e_x)

   # @jit(forceobj=True)
  def derivative(self, x : np.ndarray):
    softmax_output = self.activate(x)
    tmp = softmax_output * (np.eye(len(softmax_output)) - softmax_output)
    return np.diagonal(tmp).reshape(1, -1)
    # s = self.activate(x)
    # return np.diagonal(np.diag(s) - np.outer(s, s)).reshape(1, -1)

  def __repr__(self) -> str:
    return "Softmax"