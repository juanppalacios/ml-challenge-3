import numpy as np

class AdamOptimizer:
  def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m = None  # First moment estimate
    self.v = None  # Second raw moment estimate
    self.t = 0      # Time step

  def update(self, learning_rate, gradients, t):
    self.learning_rate = learning_rate

    if self.m is None:
      self.m = {key: np.zeros_like(value) for key, value in gradients.items()}
      self.v = {key: np.zeros_like(value) for key, value in gradients.items()}

    self.t = t

    # Update biased first moment estimate
    for key in gradients.keys():
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]

    # Update biased second raw moment estimate
    for key in gradients.keys():
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)

    # Correct bias in first moment
    m_hat = {key: self.m[key] / (1 - self.beta1 ** self.t) for key in gradients.keys()}

    # Correct bias in second raw moment
    v_hat = {key: self.v[key] / (1 - self.beta2 ** self.t) for key in gradients.keys()}

    # Update parameters
    update = {key: self.learning_rate * m_hat[key] / (np.sqrt(v_hat[key]) + self.epsilon)
              for key in gradients.keys()}

    return update
