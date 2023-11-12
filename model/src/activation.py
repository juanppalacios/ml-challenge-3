from abc import ABC, abstractmethod
import numpy as np

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

  def activate(self, x):
    return np.maximum(x, 0)

  def derivative(self, x):
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

  def activate(self, x):
    return 1 / (1 + np.exp(-x))

  def derivative(self, x):
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

  def activate(self, x):
    return np.tanh(x)

  def derivative(self, x):
    return 1 - np.tanh(x)**2

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

  def activate(self, x):
    return np.log(1 + np.exp(x))

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

  def activate(self, x):
    e_x = np.exp(x - np.max(x))  # To avoid numerical instability
    return e_x / e_x.sum()

  def derivative(self, x):
    s = self.softmax(x)
    delta = -np.outer(s, s) + np.diag(s)
    return delta

  def __repr__(self) -> str:
    return "Softmax"