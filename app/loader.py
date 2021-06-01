import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from app.utils.command import request_user_input
from app.utils.file import open_csv, open_json
from app.utils.logger import Logger
from app.meta import init_set_info, init_col_info


class DataLoader(object):
    """
    Data Loader
    """

    def __init__(self, config_path):
        self.config = open_json(config_path)

        self.filepath = self.config["dataset"]["filepath"]
        self.basepath = self.config["dataset"]["dirpath"]
        self.format = self.config["dataset"]["format"].lower()

        self.logger = Logger()
        self.dataset = self.load_dataset()  # 1.1
        self.metaset = self.load_metaset()  # 1.2

    def load_dataset(self):
        """
        1.1 Data Loading
        """
        self.logger.log(
            f"- 1.1 {self.config['dataset']['category']} type dataset loading .. ",
            level=2,
        )

        self.filepath = os.path.join(self.basepath, self.filepath)

        self.logger.log(f"- '{self.filepath}' is now loading...", level=2)

        dataset = {
            "train": self.read_csv("train"),
            "valid": self.read_csv("valid"),
            "test": self.read_csv("test"),
        }

        if dataset["test"] is None:
            dataset = self.split_dataset(dataset, "train", "test")

        return dataset

    def read_csv(self, name):
        filepath = os.path.join(self.basepath, self.filepath, name + "." + self.format)

        index_col = self.config["dataset"].get("index", None)
        index_col = index_col if index_col != "None" else None

        try:
            csv_file = open_csv(filepath=filepath, index_col=index_col)
            self.logger.log(f"- {name:5} data{csv_file.shape} is now loaded", level=3)

        except FileNotFoundError:
            csv_file = None

        return csv_file

    def split_dataset(self, dataset, origin, target):
        split_ratio = self.config["dataset"]["split_ratio"]

        dataset[origin], dataset[target] = train_test_split(
            dataset[origin], train_size=split_ratio, random_state=42
        )

        self.logger.log(
            f"- {origin:5} data{dataset[origin].shape}"
            f", {target:5} data{dataset[target].shape}"
            f"  (split ratio: {split_ratio})",
            level=3,
        )

        return dataset

    def load_metaset(self):
        """
        1.2
        """
        self.logger.log(f"- 1.2 Prepare metadata", level=2)

        def convert_dict(dtype):
            return {
                "Int64": "Num_int",
                "Float64": "Num_float",
                "object": "Cat",
            }[dtype.name]

        metaset = init_set_info(self.config, self.dataset)
        metaset = self.read_description(metaset)

        for i, col in enumerate(metaset["__columns__"]):
            col_data = self.dataset["train"][col].convert_dtypes()
            metaset[col] = init_col_info(metaset, col_data, col)

        return metaset

    def read_description(self, metaset):
        descfile = self.config["metaset"].get("descpath", None)
        if descfile is None:
            return metaset

        descpath = os.path.join(
            self.config["dataset"]["dirpath"],
            self.config["dataset"]["filepath"],
            descfile,
        )

        try:
            with open(descpath, "r", newline="\r\n") as desc_file:
                self.logger.log(f"- '{descpath}' is now loaded", level=3)

                desc_list = desc_file.read().splitlines()
                for desc_line in desc_list:
                    col, desc = desc_line.split(":")

                    metaset[col]["descript"] = desc.strip()

            return metaset

        except FileNotFoundError as e:
            self.logger.warn(f"Description File Not Found Error, '{descpath}'")
            return metaset
