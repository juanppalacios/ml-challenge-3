import pandas  as pd
import numpy   as np
import logging as log
import matplotlib.pyplot as plt
import csv

# note: consider splitting our data processor into a separate class

class Toolkit():
  def __init__(self):

    self.log_levels = {
      'DEBUG'  : log.DEBUG,
      'INFO'   : log.INFO,
      'WARNING': log.WARNING,
      'ERROR'  : log.ERROR,
    }

    self.logger    = None
    self.formatter = None
    self.handler   = None

  def configure(self, name, level = 'DEBUG'):

    assert level in self.log_levels

    self.logger    = log.getLogger(f'{name}')
    self.formatter = log.Formatter('%(name)s - %(levelname)s - %(message)s')
    self.handler   = log.StreamHandler()

    self.handler.setFormatter(self.formatter)
    self.logger.addHandler(self.handler)
    self.logger.setLevel(self.log_levels[level])

    self.debug("ready to report messages")

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
  DATA PROCESSOR
  '''
  def normalize(self, data):
    return data.astype('float32') / 255.0
    # return (data - np.min(data)) / (np.max(data) - np.min(data))

  '''
  FILE I/O
  '''

  def visualize(self, data):
    # Select a few rows (indices) that you want to reshape
    # selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    selected_indices = [0, 1, 2, 3, 4]

    # Reshape the selected rows to 28x28 images
    reshaped_images = data[selected_indices].reshape(-1, 28, 28)

    # Plot the reshaped images
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(12, 3))

    for sample, plot in enumerate(axes):
      plot.imshow(reshaped_images[sample], cmap='gray')
      plot.axis('on')

    plt.show()

  def plot(self, x, y1, y2):

    figs, axs = plt.subplots(1, 2, figsize=(12,3))

    # Plot for y1
    axs[0].plot(x, y1, label='loss')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].set_title('loss history over epochs')
    axs[0].legend()

    # Plot for y2
    axs[1].plot(x, y2, label='accuracy')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title('accuracy history over epochs')
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

  def load_data(self, path, transpose = True):
    return pd.read_csv(path).to_numpy().T if transpose else pd.read_csv(path).to_numpy()

  def save_data(self, path, data):
    # append our data to be labeled for Kaggle submission
    id_column = np.array([i+1 for i in range(data.shape[0])])
    data = np.column_stack((id_column, data))
    submission = pd.DataFrame(data, columns = ['\"Id\"','\"Expected\"']).to_csv(path, index = False, quotechar = "'")
