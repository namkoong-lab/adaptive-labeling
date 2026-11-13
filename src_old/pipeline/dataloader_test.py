import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TabularDataset(Dataset):
    def __init__(self, csv_file, y_column):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            y_column (string): Name of the column to be used as the target variable.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.y_column = y_column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Split data into features and target
        x = self.data_frame.drop(self.y_column, axis=1).iloc[idx].values
        y = self.data_frame[self.y_column].iloc[idx]

        # If idx is a list or slice, y will be DataFrames and we can use .values
        # If idx is a single value,y will be scalars, and we should not use .values
        if isinstance(idx, int):
            # Converty to 1D arrays with a single value each
            y = np.array([y])
        else:
            # Convert DataFrame to numpy array
            y = y.values

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

class TabularDatasetPool(Dataset):
    def __init__(self, csv_file, y_column):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            y_column (string): Name of the column to be used as the target variable.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.y_column = y_column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Split data into features and target
        x = self.data_frame.drop(self.y_column, axis=1).iloc[idx].values
        y = self.data_frame[self.y_column].iloc[idx]

        # If idx is a list or slice, y will be DataFrames and we can use .values
        # If idx is a single value, y will be scalars, and we should not use .values
        if isinstance(idx, int):
            # y to 1D arrays with a single value each
            y = np.array([y])
        else:
            # Convert DataFrame to numpy array
            y = y.values

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return idx, x, y

class SquareDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, idx ** 2

dataset = SquareDataset(10)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for i in range(10):
  for indices, data in dataloader:
      print(f"Indices: {indices} \t Squares: {data}")



for i in range(10):
  data_iterator = iter(dataloader)
  try:
      while True:
          indices, data = next(data_iterator)
          print(f"Indices: {indices} \t Squares: {data}")
  except StopIteration:
      # Iterator is exhausted
      print("All batches have been processed.")

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

# Generating random data for the CSV file
np.random.seed(0)  # For reproducibility
data_size = 10 # Number of rows

# Generating random features (3 features) and target values
features = np.random.rand(data_size, 1)  # Random values between 0 and 1
target = np.random.randint(0, 2, data_size)  # Random binary target

# Creating a DataFrame
df = pd.DataFrame(features, columns=['feature1'])
df['target'] = target

# Saving the DataFrame to a CSV file
csv_file_path = '/content/drive/MyDrive/example_data.csv'
df.to_csv(csv_file_path, index=False)

dataset = TabularDatasetPool(csv_file='/content/drive/MyDrive/example_data.csv', y_column='target')

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

for i in range(10):
  # Iterate over the DataLoader
  for idx, features, target in dataloader:
      print(f"Index: {idx}, Features: {features}, Target: {target}")

for i in range(10):
  data_iterator = iter(dataloader)
  try:
      while True:
          indices, x, y = next(data_iterator)
          print(f"Indices: {indices} \t Squares: {x} \t Squares: {y}")
          print(type(indices))
          print(indices.shape)
          print(x.shape)
  except StopIteration:
      # Iterator is exhausted
      print("All batches have been processed.")

dataset = TabularDatasetPool(csv_file='/content/drive/MyDrive/example_data.csv', y_column='target')

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

data_frame = pd.read_csv('/content/drive/MyDrive/example_data.csv')
y_column = 'target'

print(data_frame)

def ok(idx):
        data_frame = pd.read_csv('/content/drive/MyDrive/example_data.csv')
        y_column = 'target'
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Split data into features and target
        x = data_frame.drop(y_column, axis=1).iloc[idx].values
        print(x)
        y = data_frame[y_column].iloc[idx]
        print(y)

        z=data_frame[y_column][idx]
        print(z)

        # If idx is a list or slice, y will be DataFrames and we can use .values
        # If idx is a single value, y will be scalars, and we should not use .values
        if isinstance(idx, int):
            # y to 1D arrays with a single value each
            y = np.array([y])
        else:
            # Convert DataFrame to numpy array
            y = y.values

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return idx, x, y

ok(1)    #same if we put torch.tensor(1)

ok([1])    #same if we put torch.tensor([1])

ok([1,2])    #same if we put torch.tensor([1,2])

a=torch.tensor([1])

b=torch.tensor(1)

d=b.tolist()

d

c=torch.tensor([1,2])

ok(a)

ok(b)

l,m,n = ok(c)

l

type(l)

if torch.is_tensor(idx):
    idx = idx.tolist()

# Split data into features and target
x = data_frame.drop(y_column, axis=1).iloc[idx].values
y = data_frame[self.y_column].iloc[idx]

# If idx is a list or slice, x and y will be DataFrames and we can use .values
# If idx is a single value, x and y will be scalars, and we should not use .values
if isinstance(idx, int):
    # y to 1D arrays with a single value each
    y = np.array([y])
else:
    # Convert DataFrame to numpy array
    y = y.values

# Convert to tensor
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

return idx, x, y

