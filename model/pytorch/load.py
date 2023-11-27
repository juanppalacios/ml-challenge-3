import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self, csv_file_path_data, csv_file_path_targets, transpose_data=False, transform=None):
    self.data = pd.read_csv(csv_file_path_data).T if transpose_data else pd.read_csv(csv_file_path_data)
    self.targets = pd.read_csv(csv_file_path_targets) if csv_file_path_targets else None
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = {'data': self.data.iloc[idx].values}

    # check if targets are available
    if self.targets is not None:
      sample['target'] = self.targets.iloc[idx].values

    if self.transform:
      sample = self.transform(sample)

    return sample

class CustomDatasetFromArrays(Dataset):
  def __init__(self, data, targets, transform=None):
    self.data = torch.tensor(data, dtype=torch.float32)
    self.targets = torch.tensor(targets, dtype=torch.long)
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = {'data': self.data[idx], 'target': self.targets[idx]}

    if self.transform:
      sample = self.transform(sample)

    return sample

class ToTensor(object):
  def __call__(self, sample):
    data = sample['data']

    # check if our targets are available
    if 'target' in sample:
      target = sample['target']
      return {'data': torch.Tensor(data) / 255, 'target': torch.LongTensor(target)}
    else:
      return {'data': torch.Tensor(data) / 255}

class Normalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, sample):
    data = sample['data']

    # Check if 'target' is available
    if 'target' in sample:
      target = sample['target']
      return {
        'data': torch.div(torch.sub(torch.Tensor(data), self.mean), self.std),
        'target': torch.LongTensor(target)
      }
    else:
      return {'data': torch.div(torch.sub(torch.Tensor(data), self.mean), self.std)}


# # example usage

# TRAIN_PATH = '../data/tables/train/mnist_train.csv'
# TRAIN_TARGET_PATH = '../data/tables/train/mnist_train_targets.csv'

# TEST_PATH = '../data/tables/test/mnist_test.csv'
# # TEST_TARGET_PATH = '../data/tables/test/mnist_test_targets.csv'

# # Create instances of the dataset and dataloader
# dataset1 = CustomDataset(TRAIN_PATH, TRAIN_TARGET_PATH, transpose_data=True, transform=ToTensor())
# custom_dataloader = DataLoader(dataset1, batch_size=32, shuffle=True)

# # Iterate over the dataloader in your training loop
# for batch in custom_dataloader:
#   data, target = batch['data'], batch['target']
#   print(f"first batch: {data.shape} {target.shape}")
#   break

