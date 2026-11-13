import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(gpu: int | None = None) -> torch.device:
    """gpu=None -> cuda:0 if available else cpu; gpu=-1 -> cpu; gpu=k -> cuda:k."""
    if gpu is not None and gpu < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu if gpu is not None else 0}")
    return torch.device("cpu")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, torch.device):
            return str(obj)
        return super().default(obj)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=1)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer(store: dict | None = None, key: str | None = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if store is not None and key is not None:
        store[key] = store.get(key, 0.0) + dt


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
