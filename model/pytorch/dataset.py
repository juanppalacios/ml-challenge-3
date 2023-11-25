import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self, csv_file_path_data, csv_file_path_targets, transform=None):
    self.data = pd.read_csv(csv_file_path_data)
    self.targets = pd.read_csv(csv_file_path_targets)
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = {'data': self.data.iloc[idx].values, 'target': self.targets.iloc[idx].values}

    if self.transform:
      sample = self.transform(sample)

    return sample

# You can define a transform function if you want to apply any transformations to your data
# For example, you can convert data and targets to PyTorch tensors
# class ToTensor(object):
#   def __call__(self, sample):
#     data, target = sample['data'], sample['target']
#     return {'data': torch.Tensor(data), 'target': torch.Tensor(target)}

class ToTensor(object):
  def __call__(self, sample):
    data, target = sample['data'], sample['target']
    return {'data': torch.Tensor(data), 'target': torch.LongTensor(target)}

# # # Example usage:
# # Define your file paths
# csv_file_path_data = 'path/to/training_data.csv'
# csv_file_path_targets = 'path/to/training_targets.csv'

# # Create instances of the dataset and dataloader
# custom_dataset = CustomDataset(csv_file_path_data, csv_file_path_targets, transform=ToTensor())
# custom_dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

# # Iterate over the dataloader in your training loop
# for batch in custom_dataloader:
#   data, target = batch['data'], batch['target']
#   # Your training code here
