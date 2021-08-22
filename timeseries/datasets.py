import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from timeseries.logger import Logger

logger = Logger(__file__)

class TimeseriesDataset:
    def __init__(self, config, device):
        # Load Data
        self.config = config
        self.device = device
        # Dataset option
        self.title = config["data"]
        self.label = config["anomaly"] or None
        self.workers = config["workers"]

        self.stride = config["stride"]
        self.seq_len = config["seq_len"]
        self.shuffle = config["shuffle"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]

        # Dataset
        data, label = self.load_data(config)

        self.n_feature = len(data.columns)
        self.time = self.store_times(data)
        self.data = self.store_values(data, normalize=True)
        self.label = self.store_values(label, normalize=False)
        
        self.data_len = len(self.data)
        

    def load_data(self, config):
        path = os.path.join(config["path"], config["data"])
        data = pd.read_csv(path)

        INDEX_NOT_FOUND_ERR = f"Column `{config['index']}` is not found"
        assert config["index"] in data.columns, INDEX_NOT_FOUND_ERR
        data[config["index"]] = pd.to_datetime(data[config["index"]])
        label = self.load_anomaly(config, data)

        data = data.set_index(config["index"])
        return data, label

    def load_anomaly(self, config, data):
        if self.label is None:
            logger.info("Anomaly labels were not provided")
            return None

        path = os.path.join(config["path"], config["anomaly"])
        if not os.path.exists(path):
            logger.info(f"Anomaly file path is not corrected : {path}")
            return None

        with open(path) as f:
            anomalies = json.load(f)["anomalies"]
        
        label = np.zeros(len(data))
        for ano_span in anomalies:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in data.index:
                if data.loc[idx, config["index"]] >= ano_start and data.loc[idx, config["index"]] <= ano_end:
                    label[idx] = 1.0
        
        label = pd.DataFrame({config["index"] : data[config["index"]], "value": label})
        label = label.set_index(config["index"])
        
        return label

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]

    def store_times(self, data):
        time = pd.to_datetime(data.index)
        time = time.strftime("%y%m%d:%H%M")
        time = time.values
        return time

    def store_values(self, data, normalize=False):
        data = self.windowing(data[["value"]])
        if normalize is True:
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
