import os
import yaml
import json
import torch
import numpy as np
import pandas as pd

class TimeseriesDataset:
    def __init__(self, config, device):
        # Load Data
        self.config = config
        self.device = device
        # Dataset option
        self.title = config["dataset"]["key"]
        self.workers = config["workers"]

        self.stride = config["stride"]
        self.seq_len = config["seq_len"]
        self.shuffle = config["shuffle"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]

        # Dataset
        data = self.load_data(config["dataset"])
        self.n_feature = len(data.columns)
        self.time = self.store_times(data)
        self.data = self.store_values(data)
        self.data_len = len(self.data)
        
        # data_np = self.data.numpy().reshape(self.data_len, -1)
        # data_df = pd.DataFrame(data_np)
        # data_df.to_csv(self.title)

    def load_data(self, config):
        path = os.path.join(config["path"], config["key"])
        data = pd.read_csv(path)

        INDEX_NOT_FOUND_ERR = f"Column `{config['index']}` is not found"
        assert config["index"] in data.columns, INDEX_NOT_FOUND_ERR
        data = data.set_index(config["index"])

        return data

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]

    def store_times(self, data):
        time = pd.to_datetime(data.index)
        time = time.strftime("%y%m%d_%H%M")
        time = self.windowing(time)
        return time

    def store_values(self, data):
        data = self.windowing(data[["value"]])
        data = self.normalize(data)
        data = torch.from_numpy(data).float()
        return data

    def windowing(self, x):
        stop = len(x) - self.seq_len
        return np.array([x[i : i + self.seq_len] for i in range(0, stop, self.stride)])

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()

        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception(
                "You are calling denormalize, but the input was not normalized"
            )
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)

    def get_samples(self, netG, shape, cond):
        idx = np.random.randint(self.data.shape[0], size=shape[0])
        x = self.data[idx].to(self.device)
        z = torch.randn(shape).to(self.device)
        if cond > 0:
            z[:, :cond, :] = x[:, :cond, :]

        y = netG(z).to(self.device)

        return y, x
