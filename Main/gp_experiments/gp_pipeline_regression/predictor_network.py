import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


class PredictorNetwork(nn.Module):
    """
    A simple MLP predictor network for regression.

    Input: x of shape [batch_size, input_dim]
    Output: predictions of shape [batch_size, output_size]
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_size=1, dropout=0.1):
        super(PredictorNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_size = output_size

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, input_dim]
        # output: [batch_size, output_size]
        return self.network(x)


class ZeroPredictor(nn.Module):
    """
    A constant zero predictor for baseline comparisons.
    Always returns zeros regardless of input.
    Output shape: [batch_size] (1D tensor)
    """
    def __init__(self):
        super(ZeroPredictor, self).__init__()
        self.input_dim = None  # Compatible with any input dim
        self.output_size = 1

    def forward(self, x):
        # x: [batch_size, input_dim]
        # output: [batch_size] of zeros
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    def eval(self):
        # No-op, always in eval mode
        return self


class ConstantValueNetwork(nn.Module):
    """
    A constant value predictor (exact copy from constant_network.py).
    Returns the same constant value for all inputs.
    Output shape: [batch_size, output_size] (2D tensor)

    NOTE: Default constant_value is 1.0, NOT 0.0!
    Use constant_value=0.0 explicitly for zero baseline.
    """
    def __init__(self, constant_value=1.0, output_size=1):
        super(ConstantValueNetwork, self).__init__()
        # Define the constant value and output size
        self.constant_value = nn.Parameter(torch.tensor([constant_value]*output_size), requires_grad=False)
        self.output_size = output_size
        self.input_dim = None  # Compatible with any input dim

    def forward(self, x):
        # x is your input tensor. Its value is ignored in this model.
        # Return a 2-D tensor with the constant value for each item in the batch.
        batch_size = x.size(0)  # Get the batch size from the input
        return self.constant_value.expand(batch_size, self.output_size)


def get_zero_predictor(device='cpu'):
    """
    Get a zero predictor instance.

    Args:
        device: Device (ignored, but kept for API compatibility)

    Returns:
        ZeroPredictor instance
    """
    print("Using zero predictor (constant baseline)")
    return ZeroPredictor()


def train_predictor(
    train_x,
    train_y,
    device,
    input_dim=None,
    hidden_dims=[64, 32],
    output_size=1,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=100,
    batch_size=32,
    val_split=0.2,
    early_stopping_patience=10,
    verbose=False
):
    """
    Train a predictor network on the given training data.

    Args:
        train_x: Training features, shape [N, input_dim], torch.Tensor
        train_y: Training targets, shape [N], torch.Tensor
        device: torch device to use
        input_dim: Input dimension (inferred from train_x if None)
        hidden_dims: List of hidden layer dimensions
        output_size: Output dimension (default 1 for regression)
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        n_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        early_stopping_patience: Number of epochs to wait for improvement
        verbose: Whether to print training progress

    Returns:
        model: Trained PredictorNetwork in eval mode
    """
    # Infer input dimension if not provided
    if input_dim is None:
        input_dim = train_x.size(1)

    # Ensure data is on CPU for splitting
    train_x_cpu = train_x.detach().cpu()
    train_y_cpu = train_y.detach().cpu()

    # Split data into train and validation sets
    n_samples = train_x_cpu.size(0)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Shuffle indices
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    x_train = train_x_cpu[train_indices].to(device)
    y_train = train_y_cpu[train_indices].to(device)
    x_val = train_x_cpu[val_indices].to(device)
    y_val = train_y_cpu[val_indices].to(device)

    # Ensure y has correct shape for loss computation
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)  # [N] -> [N, 1]
    if y_val.dim() == 1:
        y_val = y_val.unsqueeze(1)  # [N] -> [N, 1]

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = PredictorNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= n_train

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val)
            val_loss = criterion(val_predictions, y_val).item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")

    return model


def save_predictor(model, filepath, input_dim=None, hidden_dims=None, output_size=None, dropout=None):
    """
    Save a trained predictor network to disk.

    Args:
        model: Trained PredictorNetwork
        filepath: Path to save the model (e.g., 'predictor.pt')
        input_dim: Input dimension (optional, for metadata)
        hidden_dims: Hidden layer dimensions (optional, for metadata)
        output_size: Output dimension (optional, for metadata)
        dropout: Dropout rate (optional, for metadata)
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim if input_dim is not None else model.input_dim,
        'output_size': output_size if output_size is not None else model.output_size,
    }

    # Save architecture info if provided
    if hidden_dims is not None:
        save_dict['hidden_dims'] = hidden_dims
    if dropout is not None:
        save_dict['dropout'] = dropout

    torch.save(save_dict, filepath)
    print(f"Predictor saved to {filepath}")


def load_predictor(filepath, device='cpu', hidden_dims=[64, 32], dropout=0.1):
    """
    Load a trained predictor network from disk.

    Args:
        filepath: Path to the saved model, or special values:
            - "zero": ZeroPredictor (returns [batch_size] of zeros)
            - "const:VALUE": ConstantValueNetwork with specified value
                e.g., "const:0.0" for zero, "const:1.0" for ones
        device: Device to load the model to
        hidden_dims: Hidden layer dimensions (used if not saved in checkpoint)
        dropout: Dropout rate (used if not saved in checkpoint)

    Returns:
        model: Loaded PredictorNetwork in eval mode, or constant predictor
    """
    # Special case: zero predictor
    if filepath.lower() == "zero":
        return get_zero_predictor(device)

    # Special case: constant value predictor (e.g., "const:0.0" or "const:1.0")
    if filepath.lower().startswith("const:"):
        try:
            const_value = float(filepath.split(":")[1])
            print(f"Using ConstantValueNetwork with constant_value={const_value}")
            model = ConstantValueNetwork(constant_value=const_value, output_size=1).to(device)
            model.eval()
            return model
        except (ValueError, IndexError):
            raise ValueError(f"Invalid constant format: {filepath}. Use 'const:VALUE' e.g., 'const:0.0'")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictor file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    input_dim = checkpoint['input_dim']
    output_size = checkpoint['output_size']

    # Use saved architecture if available, otherwise use defaults
    if 'hidden_dims' in checkpoint:
        hidden_dims = checkpoint['hidden_dims']
    if 'dropout' in checkpoint:
        dropout = checkpoint['dropout']

    model = PredictorNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Predictor loaded from {filepath}")
    return model
