import os
from app.utils.file import open_json, open_csv
from app.utils.logger import Logger
from app.utils.command import request_user_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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
        self.dataset = self.load_dataset()
        self.metaset = self.load_metaset()

    def load_dataset(self):
        self.logger.log(f"- {self.config['dataset']['category']} type dataset loading .. ", level=2)

        #if self.format == "csv" TODO : Fill some other type of data
        filepath = os.path.join(self.basepath, self.filepath + "." + self.format)

        self.logger.log(f"- '{filepath}' is now loading...", level=2)

        dataset = {
            'train': self.read_csv('train'),
            'valid': self.read_csv('valid'),
            'test': self.read_csv('test')
        }

        if dataset['test'] is None:
            dataset = self.split_dataset(dataset, 'train', 'test')

        return dataset

    def read_csv(self, name):
        filepath = os.path.join(self.basepath, self.filepath, name + "." + self.format)
        
        index_col = self.config["dataset"].get('index', None) 
        index_col = index_col if index_col != 'None' else None

        try:
            csv_file = open_csv(filepath=filepath, index_col=index_col)
            self.logger.log(f"- {name:5} data{csv_file.shape} is now loaded", level=3)

        except FileNotFoundError:
            csv_file = None

        return csv_file

    def split_dataset(self, dataset, origin, target):
        split_ratio = self.config['dataset']['split_ratio']

        dataset[origin], dataset[target] = train_test_split(
            dataset[origin], train_size=split_ratio, random_state=42)

        self.logger.log(
            f"- {origin:5} data{dataset[origin].shape}"
            f", {target:5} data{dataset[target].shape}"
            f"  (split ratio: {split_ratio})", level=3)

        return dataset

    
    def load_metaset(self):
        self.logger.log(f"- 1.2 Prepare metadata", level=2)

        def convert_dict(dtype):
            return {
                'Int64': 'Num_int',
                'Float64': 'Num_float',
                'object': 'Cat',
            }[dtype.name]

        def distribute(target, name):
            values = dataset[name][target].value_counts()
            length = metaset["__nrows__"][name]
            return round(values/length*100, 3).to_frame(name=name)


        dataset = self.dataset

        metaset = dict()

        trainset, testset = dataset['train'], dataset['test']
        train_col, test_col = trainset.columns, testset.columns

        target_label = self.config["dataset"].get("target_label", train_col.difference(test_col).values)
        
        metaset["__target__"] = target_label
        metaset["__nrows__"] = {"train": len(trainset), "test": len(testset)}
        metaset["__ncolumns__"] = len(train_col)
        metaset["__columns__"] = pd.Series(train_col.values)
        metaset['__distribution__'] = pd.concat(
            [distribute(target_label, 'train'), distribute(target_label, 'test')],
            axis=1, names=["train", "test"]
        )

        for i, col in enumerate(metaset["__columns__"]):
            col_data = trainset[col].convert_dtypes()
            col_meta = {
                "index": i,
                "name": col,
                "dtype": str(col_data.dtype),
                "descript": None,
                "nunique": col_data.nunique(),
                "na_count": col_data.isna().sum(),
                "target": (metaset["__target__"] == col),
                "log": list()
            }

            if col_meta["dtype"][:3] == "Cat":
                col_meta["stat"] = {
                    "unique": col_data.unique(),
                }
                col_meta["dtype"] = f"{col_meta['dtype']}_{col_meta['nunique']}"
            elif col_meta["dtype"] == "Int64" or col_meta["dtype"] == "Float64":
                col_meta["stat"] = {
                    "skew": round(col_data.skew(), 4),
                    "kurt": round(col_data.kurt(), 4),
                    "unique": col_data.unique(),
                }

            metaset[col] = col_meta
        
        metaset = self.read_description(metaset)

        return metaset

    def read_description(self, metaset):
        descfile = self.config["metaset"].get("descpath", None)
        if descfile is None:
            return

        descpath = os.path.join(
            self.config["dataset"]["dirpath"],
            self.config["dataset"]["filepath"], descfile
        )
        
        try:
            with open(descpath, "r", newline="\r\n") as desc_file:
                self.logger.log(f"- '{descpath}' is now loaded", level=3)

                desc_list = desc_file.read().splitlines()
                for desc_line in desc_list:
                    col, desc = desc_line.split(":")
                    
                    metaset[col]['descript'] = desc.strip()
                
            return metaset
                

        except FileNotFoundError as e:
            self.logger.warn(f"Description File Not Found Error, '{descpath}'")
            return None
