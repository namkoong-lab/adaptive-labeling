import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Returns features with shape [Batch_size, features], Labels with shape [Batch_size,1]

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
        x = self.data_frame.drop(self.y_column, axis=1).iloc[idx]
        y = self.data_frame[self.y_column].iloc[idx]

        # If idx is a list or slice, y will be DataFrames and we can use .values
        # If idx is a single value, y will be scalars, and we should not use .values
        if isinstance(idx, int):
            # Convert y to 1D arrays with a single value each
            y = np.array([y])
        else:
            # Convert DataFrame to numpy array
            y = y.values

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

# Returns features with shape [Batch_size, features], Labels with shope [Batch_size,1], Indices with shape [Batch_size] --- required for extracting weights of a given batch of pool

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