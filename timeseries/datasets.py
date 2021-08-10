import os
import yaml
import json
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class DataSettings:
    def __init__(self, file, train=None):
        with open(file) as f:
            config = yaml.safe_load(f)["data"]

        self.end_name = config["end_name"]

        self.train = train
        self.window_length = config["window_length"]

        self.BASE = config["BASE"]
        self.label_file = config["label"]
        self.data_file = config["data"] + self.end_name
        self.key = config["key"] + self.end_name
        self.vis_opt = config["visualize"]
        self.gap_opt = config["fill_timegap"]

    def load_ano_file(self):
        file = os.path.join(self.BASE, self.label_file)

        with open(file) as f:
            json_label = json.load(f)

        return json_label[self.key]

    def load_csv_file(self):
        file = os.path.join(self.BASE, self.data_file)
        data = pd.read_csv(file)
        data = data.set_index("timestamp")

        return data


class Dataset(Dataset):
    def __init__(self, settings):
        """
        Args:
            settings (object): settings for loading data and preprocessing
        """
        self.dataset = settings.end_name

        self.train = settings.train
        self.gap_opt = settings.gap_opt
        self.seq_len = settings.window_length
        self.stride = 4

        data = settings.load_csv_file()
        self.times = self.store_times(data)
        self.data = self._preprocessing(data)

        self.data_len = self.data.shape[0]
        self.n_feature = self.data.shape[2]

    def _preprocessing(self, x):
        data = self.windowing(x[["value"]])
        data = torch.from_numpy(data).float()
        data = self.normalize(data)
        return data

    def store_times(self, data):
        time = pd.to_datetime(data.index)
        time = time.strftime("%y%m%d_%H%M")
        time = self.windowing(time)
        return time

    def windowing(self, x):
        stop = len(x) - self.seq_len
        return np.array([x[i : i + self.seq_len] for i in range(0, stop, self.stride)])

    def get_samples(self, netG, shape, cond, device):
        idx = np.random.randint(self.data.shape[0], size=shape[0])
        x = self.data[idx].to(device)
        z = torch.randn(shape).to(device)
        if cond > 0:
            z[:, :cond, :] = x[:, :cond, :]

        y = netG(z)

        # y = torch.cat((x.float(), y.float()), dim=2)
        y = y.to(device)

        return y, x

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]

    def get_prev(self, idx):
        if idx == 0:
            return self.data[idx]

        return self.data[idx - 1]

    def get_dayofweek(self, idx):
        return self.date[idx].weekday()

    def assign_ano(self, df_x=None):
        y = np.zeros(len(df_x))

        for ano_span in self.ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])

            for idx in df_x.index:
                if (
                    df_x.loc[idx, "timestamp"] >= ano_start
                    and df_x.loc[idx, "timestamp"] <= ano_end
                ):
                    y[idx] = 1.0

        df_y = pd.DataFrame(y)

        return df_x, df_y

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

    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return (self.delta_max - self.delta_min) * (x - self.or_delta_min) / (
            self.or_delta_max - self.or_delta_min
        ) + self.delta_min
