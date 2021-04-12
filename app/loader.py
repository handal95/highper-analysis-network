import os
from app.utils.file import open_json, open_csv
from app.utils.logger import Logger
from app.utils.command import request_user_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
                'int64': 'Num_int',
                'float64': 'Num_float',
                'object': 'Cat',
            }[dtype.name]

        desc_path = self.config.get("DESCRIPT_PATH", None)
        dataset = self.dataset

        metaset = dict()

        trainset, testset = dataset['train'], dataset['test']
        train_col, test_col = trainset.columns, testset.columns

        target_label = self.config["dataset"].get("target_label", train_col.difference(test_col).values)
        
        metaset["__target__"] = target_label
        metaset["__nrows__"] = {"train": len(trainset), "test": len(testset)}
        metaset["__ncolumns__"] = len(train_col)
        metaset["__columns__"] = train_col.values
        train_distribution = dataset['train'][target_label].value_counts()
        test_distribution = dataset['test'][target_label].value_counts()
        metaset["__distribution__"] = {
            'train': (train_distribution/len(trainset))*100,
            'test': (test_distribution/len(testset))*100,
        }

        for i, col in enumerate(metaset["__columns__"]):
            col_data = trainset[col]
            col_meta = {
                "index": i,
                "name": col,
                "dtype": convert_dict(col_data.dtype),
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
            elif col_meta["dtype"][:3] == "Num":
                col_meta["stat"] = {
                    "skew": round(col_data.skew(), 4),
                    "kurt": round(col_data.kurt(), 4),
                    "unique": col_data.unique(),
                }

            metaset[col] = col_meta

        return metaset
