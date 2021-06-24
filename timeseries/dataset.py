from numpy.core.numeric import full
import yaml
from torch.utils.data import Dataset
import numpy as np
import json
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import torch
from pathlib import Path
import matplotlib.animation as animation


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

class NabDataset(Dataset):
    def __init__(self, data_settings):
        """
        Args:
            data_settings (object): settings for loading data and preprocessing
        """

        self.dataset = data_settings.end_name
        self.train = data_settings.train
        self.gap_opt = data_settings.gap_opt
        # self.ano_span_count is updated
        #     when read_data() function is called
        self.ano_span_count = 0
        self.window_length = data_settings.window_length

        df_x, df_y = self.read_data(
            data_file=data_settings.data_file,
            label_file=data_settings.label_file,
            key=data_settings.key,
            BASE=data_settings.BASE,
        )

        # select and standardize data
        # if data_settings.vis_opt:
        #     self.visualize(df_x, df_y)

        df_times = df_x["timestamp"]
        df_x = df_x[["value"]]
        df_x = self.normalize(df_x)
        df_x.columns = ["value"]

        df_x = df_x[:256]
        df_y = df_y[:256]
        
        if data_settings.vis_opt and self.train:
            self.normalized_visualize(df_times, df_x, df_y)


        # important parameters
        # self.window_length = int(len(df_x)*0.1/self.ano_span_count)
        if data_settings.train:
            self.stride = 1
        else:
            self.stride = self.window_length

        self.n_feature = len(df_x.columns)

        # x, y data
        x = df_x
        y = df_y

        # adapt the datasets for the sequence data shape
        x, y = self.unroll(x, y)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(
            np.array([1 if sum(y_i) > 0 else 0 for y_i in y])
        ).float()

        self.data_len = x.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # create sequences
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)

        idx = 0
        while idx < len(data) - seq_len:
            un_data.append(data.iloc[idx : idx + seq_len].values)
            un_labels.append(labels.iloc[idx : idx + seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)

    def read_data(self, data_file=None, label_file=None, key=None, BASE=""):
        with open(BASE + label_file) as FI:
            j_label = json.load(FI)
        ano_spans = j_label[key]
        self.ano_span_count = len(ano_spans)
        df_x = pd.read_csv(BASE + data_file)
        df_x["timestamp"] = pd.to_datetime(df_x["timestamp"])

        if self.gap_opt:
            df_x, gap_y = self.fill_timegap(df_x)
            # df_x.to_csv("./fill_timegap.csv")

        df_x, df_y = self.assign_ano(ano_spans, df_x)
        
        if self.gap_opt:
            df_y = self.fill_anogap(df_y, gap_y)
            
        return df_x, df_y

    def assign_ano(self, ano_spans=None, df_x=None):
        y = np.zeros(len(df_x))
        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if (
                    df_x.loc[idx, "timestamp"] >= ano_start
                    and df_x.loc[idx, "timestamp"] <= ano_end
                ):
                    y[idx] = 1.0
                
        return df_x, pd.DataFrame(y)

    def normalize(self, df_x=None):

        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x


    def fill_timegap(self, df_x):
        TIMEGAP = df_x["timestamp"][1] - df_x["timestamp"][0]

        new_df_x = pd.DataFrame()
        gap_y = list()

        for i in range(df_x.shape[0] - 1):
            timegap = df_x["timestamp"][i+1] - df_x["timestamp"][i]
            if timegap != TIMEGAP:
                for gap in range(timegap//TIMEGAP):
                    new_row = {
                        "timestamp": df_x["timestamp"][i] + gap * TIMEGAP,
                        "value": df_x["value"][i]
                    }
                    gap_y.append(len(new_df_x))
                    new_df_x = new_df_x.append(new_row, ignore_index=True)
            else:
                new_df_x = new_df_x.append(
                    df_x.iloc[i], ignore_index=True)
        return new_df_x, gap_y
    
    def fill_anogap(self, df_y, gap_y=None):
        for gap in gap_y:
            df_y.loc[gap] = 1.0
        
        return df_y
        

    def visualize(self, data, ano):
        fig = plt.figure(figsize=(10, 10))
        plt.ion()

        y = data["value"]
        x = data["timestamp"]

        fulldata_x = x.values.reshape(1, -1)
        fulldata_y = y.values.reshape(1, -1)
        fullanomal = ano.values.reshape(1, -1)
        
        time_pivot = 240

        data_x = fulldata_x[:, :time_pivot]
        data_y = fulldata_y[:, :time_pivot]
        ano_y = fullanomal[:, :time_pivot]
        
        fig = plt.figure(figsize=(16, 8))
        ax = plt.subplot(111, frameon=False)
        plt.grid(True)

        lines = []
        for i in range(fulldata_x.shape[0]):
            lw = 1 - 2 * i / 20.0
            line, = ax.plot(data_x[i], data_y[i], "b-", color="k", lw=lw)
            lines.append(line)
            
        ax.set_xlabel("Time")
        ax.set_ylabel("Data")
        ax.text(0.4, 1.0, self.dataset, transform=ax.transAxes, ha="right", va="bottom", color="k", 
                family="sans-serif", fontweight="bold", fontsize=16)
        ax.set_title("normal", loc="right", color="b", fontsize=16)

        def update(*args):
            fulldata_x[:, :-1] = fulldata_x[:, 1:]
            fulldata_y[0, :-1] = fulldata_y[:, 1:]
            fullanomal[:, :-1] = fullanomal[:, 1:]

            data_x[:, :] = fulldata_x[:, :time_pivot]
            data_y[:, :] = fulldata_y[:, :time_pivot]
            ano_y[:, :] = fullanomal[:, :time_pivot]
            
            if data_x[:, -1] == fulldata_x[:, -1]:
                plt.clf()
                plt.close()

            for i in range(len(data_y)):
                convert = ano_y[:, -2] - ano_y[:, -1] 
                if convert > 0:
                    ax.set_title("normal", loc="right", color="b", fontsize=16)
                elif convert < 0:
                    ax.set_title("Anomalies Detected", loc="right", color="r", fontweight="bold", fontsize=16)
                    
                lines[i].set_xdata(data_x[i])
                lines[i].set_ydata(data_y[i])
                ax.relim()
                ax.autoscale_view()

        anim = animation.FuncAnimation(fig, update, interval=1)
        

    def normalized_visualize(self, time, value, ano):
        fig = plt.figure(figsize=(10, 10))
        plt.ion()

        y = value
        x = time

        fulldata_x = x.values.reshape(1, -1)
        fulldata_y = y.values.reshape(1, -1)
        fullanomal = ano.values.reshape(1, -1)
        
        time_pivot = 240

        data_x = fulldata_x[:, :time_pivot]
        data_y = fulldata_y[:, :time_pivot]
        ano_y = fullanomal[:, :time_pivot]
        
        fig = plt.figure(figsize=(16, 8))
        ax = plt.subplot(111, frameon=False)
        plt.grid(True)

        lines = []
        for i in range(fulldata_x.shape[0]):
            lw = 1 - 2 * i / 20.0
            line, = ax.plot(data_x[i], data_y[i], "b-", color="k", lw=lw)
            lines.append(line)

        plt.ylim([-3, 3])
        ax.set_xlabel("Time")
        ax.set_ylabel("Data")
        ax.text(0.4, 1.0, self.dataset, transform=ax.transAxes, ha="right", va="bottom", color="k", 
                family="sans-serif", fontweight="bold", fontsize=16)
        ax.set_title("normal", loc="right", color="b", fontsize=16)

        def update(*args):
            fulldata_x[:, :-1] = fulldata_x[:, 1:]
            fulldata_y[0, :-1] = fulldata_y[:, 1:]
            fullanomal[:, :-1] = fullanomal[:, 1:]

            data_x[:, :] = fulldata_x[:, :time_pivot]
            data_y[:, :] = fulldata_y[:, :time_pivot]
            ano_y[:, :] = fullanomal[:, :time_pivot]
            
            if data_x[:, -1] == fulldata_x[:, -1]:
                plt.clf()
                plt.close()

            for i in range(len(data_y)):
                convert = ano_y[:, -2] - ano_y[:, -1] 
                if convert > 0:
                    ax.set_title("normal", loc="right", color="b", fontsize=16)
                elif convert < 0:
                    ax.set_title("Anomalies Detected", loc="right", color="r", fontweight="bold", fontsize=16)
                    
                lines[i].set_xdata(data_x[i])
                lines[i].set_ydata(data_y[i])

                ax.relim()
                ax.autoscale_view()
        anim = animation.FuncAnimation(fig, update, interval=1)
        input()