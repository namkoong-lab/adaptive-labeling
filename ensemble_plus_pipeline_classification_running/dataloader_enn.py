import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import pandas as pd
import numpy as np

# Returns features with shape [Batch_size, features], Labels with shape [Batch_size,1]


class BootstrappedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Sample with replacement using indices
        indices = torch.randint(0, len(self.data_source), (len(self.data_source),)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)



class TabularDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]
    
    def update_targets(self, new_y):
        """
        Update the targets tensor.
        
        Args:
            new_y (Tensor): The new targets tensor.
        """
        self.y = new_y
    

class TabularDatasetPool(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        return idx, self.x[idx], self.y[idx]

    def update_targets(self, new_y):
        """
        Update the targets tensor.
        
        Args:
            new_y (Tensor): The new targets tensor.
        """
        self.y = new_y  



class TabularDatasetCsv(Dataset):
    def __init__(self, csv_file, y_column):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            y_column (string): Name of the column to be used as the target variable.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.y_column = y_column
        
        self.x = torch.tensor(self.data_frame.drop(self.y_column, axis=1).values, dtype=torch.float32)        
        self.y = torch.tensor(self.data_frame[self.y_column].values, dtype=torch.float32)
        
        # Split data into features and target
        #x = torch.tensor(self.data_frame.iloc[:, :-1].values, dtype=torch.float32)
        #self.y = torch.tensor(self.data_frame[self.y_column].reshape(-1, 1).values, dtype=torch.float32)
        # If idx is a list or slice, y will be DataFrames and we can use .values
        # If idx is a single value, y will be scalars, and we should not use .values
        #if isinstance(idx, int):
            # Convert y to 1D arrays with a single value each
        #    y = np.array([y])
        #else:
            # Convert DataFrame to numpy array
         #   y = y.values
        # Convert to tensor
        #x = torch.tensor(x, dtype=torch.float32)
        #y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]

# Returns features with shape [Batch_size, features], Labels with shope [Batch_size,1], Indices with shape [Batch_size] --- required for extracting weights of a given batch of pool

class TabularDatasetPoolCsv(Dataset):
    def __init__(self, csv_file, y_column):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            y_column (string): Name of the column to be used as the target variable.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.y_column = y_column
        # Split data into features and target
        self.x = torch.tensor(self.data_frame.drop(self.y_column, axis=1).values, dtype=torch.float32)
        #x = torch.tensor(self.data_frame.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(self.data_frame[self.y_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        return idx, self.x[idx], self.y[idx]