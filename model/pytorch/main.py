from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import datetime

# note: used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# note: custom imports for data set
from load import CustomDataset, CustomDatasetFromArrays, ToTensor, Normalize

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc0 = nn.Linear(28 * 28, 512)
    self.bn0 = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(512, 256)
    self.bn1 = nn.BatchNorm1d(256)
    self.fc2 = nn.Linear(256, 128)
    self.bn2 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(128, 64)
    self.bn3 = nn.BatchNorm1d(64)
    self.fc4 = nn.Linear(64, 32)
    self.bn4 = nn.BatchNorm1d(32)
    self.fc5 = nn.Linear(32, 16)
    self.bn5 = nn.BatchNorm1d(16)
    self.fc6 = nn.Linear(16, 10)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = F.relu(self.bn0(self.fc0(x)))
    x = self.dropout(x)
    x = F.relu(self.bn1(self.fc1(x)))
    x = self.dropout(x)
    x = F.relu(self.bn2(self.fc2(x)))
    x = self.dropout(x)
    x = F.relu(self.bn3(self.fc3(x)))
    x = self.dropout(x)
    x = F.relu(self.bn4(self.fc4(x)))
    x = self.dropout(x)
    x = F.relu(self.bn5(self.fc5(x)))
    output = x # F.log_softmax(x, dim=1)

    return output

def train(args, model, device, train_loader, optimizer, epoch):
  model.train()

  train_loss_history = []

  for batch_idx, batch in enumerate(train_loader):
    data, target = batch['data'].to(device), torch.flatten(batch['target']).to(device)
    optimizer.zero_grad()
    output = model(data)
    # loss = F.nll_loss(output, target)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      if args.dry_run:
        break
    train_loss_history.append(loss.item())

  return np.mean(train_loss_history)


def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  test_accuracy_history = []

  with torch.no_grad():
    for batch in test_loader:
      data, target = batch['data'].to(device), batch['target'].to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  test_accuracy_history.append(100. * correct / len(test_loader.dataset))

  return test_accuracy_history

def predict(model, device, test_loader):
  model.eval()
  predictions = []
  global print_index

  with torch.no_grad():
    for data in test_loader:
      data = data['data'].to(device)
      output = model(data)
      pred = output.argmax(dim = 1)
      predictions.extend(pred.cpu().numpy())
  return predictions

def plot_over_epochs(epochs, data, description):
  current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  plt.plot(range(1, len(data) + 1), data, marker='o')
  plt.title(description)
  plt.xlabel('Epoch')
  if 'accuracy' in description:
    plt.ylabel('Accuracy')
    plt.savefig(f'./performance/acc_{current_time}.png')
  else:
    plt.ylabel('Loss')
    plt.savefig(f'./performance/loss_{current_time}.png')
  plt.close()

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=14, metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=True,
                      help='disables CUDA training')
  parser.add_argument('--no-mps', action='store_true', default=True,
                      help='disables macOS GPU training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=True,
                      help='For Saving the current Model')

  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  use_mps = not args.no_mps and torch.backends.mps.is_available()

  # setting device on GPU if available, else CPU
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  torch.manual_seed(args.seed)

  if use_cuda:
    device = torch.device("cuda")
  elif use_mps:
    device = torch.device("mps")
  else:
    device = torch.device("cpu")

  print(f"using {device} device...")

  train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
  test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
  # if use_cuda:
  cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)

  TRAIN_PATH = '../data/train/mnist_train.csv'
  TRAIN_TARGET_PATH = '../data/train/mnist_train_targets.csv'

  TEST_PATH = '../data/test/mnist_test.csv'
  TEST_TARGET_PATH = '../data/test/mnist_test_targets.csv'

  transform = ToTensor()

  # Create instances of the dataset and dataloader
  dataset = CustomDataset(TRAIN_PATH, TRAIN_TARGET_PATH, transpose_data = True, transform = transform)
  dataset2 = CustomDataset(TEST_PATH, None, transpose_data = True,  transform = transform) # note: using cross validation data

  train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  model = Net().to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
  # optimizer = optim.SGD(model.parameters(), lr=args.lr)

  acc_history  = []
  loss_history = []

  #> cross-validation setup
  cross_validate = True
  if cross_validate:
    n_splits = 5
    features, labels = dataset.data.values, dataset.targets.values.flatten()

    # define cross-validation strategy
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    for fold, (train_index, val_index) in enumerate(skf.split(features, labels)):
      print(f"\nCross Validation Fold {fold + 1}/{n_splits}:")

      #> split our data for cross-validation train/test
      train_data, val_data = features[train_index], features[val_index]
      train_targets, val_targets = labels[train_index], labels[val_index]

      # Create DataLoader instances for training and validation sets
      train_dataset = CustomDatasetFromArrays(train_data, train_targets, transform=transform)
      val_dataset = CustomDatasetFromArrays(val_data, val_targets, transform=transform)

      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

      scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
      for epoch in range(1, args.epochs + 1):
        loss_history.append(train(args, model, device, train_loader, optimizer, epoch))
        acc_history.append(test(model, device, val_loader)) # note: using our test cross validation data
        scheduler.step()

    plot_over_epochs(epoch, acc_history, description='accuracy history over epochs')
    plot_over_epochs(epoch, loss_history, description='loss history over epochs')

  #> Kaggle-submission setup

  # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  # for epoch in range(1, args.epochs + 1):
  #   train(args, model, device, train_loader, optimizer, epoch)
  #   # test(model, device, test_loader) # note: using our test cross validation data
  #   scheduler.step()

  # After training on cross-validation, get predictions for the test set
  test_predictions = predict(model, device, test_loader)

  # Save or process the test_predictions as needed for Kaggle submission
  print("printing our predictions to our .csv file!")
  submission_df = pd.DataFrame({"Id": range(1, len(test_predictions) + 1), "Expected": test_predictions})
  submission_df.to_csv('../data/test/pytorch_mnist_test_targets.csv', index=False)

  # note: [3, 4, 1, 1, 4, 1, 8, 9, 1, 3]

  # if args.save_model:
  #   torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()