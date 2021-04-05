import os
import pandas as pd
import numpy as np
import pprint
from app.utils.logger import Logger


logger = Logger()


def prepare_data(config):
    logger.log(f"- 1.1 Prepare dataset", level=2)

    if config['DATASET_TYPE'] == 'CSV':
        logger.log(f"- '{config['DATASET']}.csv' is now loading...", level=3)
        file_path = config['DATASET_PATH']
        dataset = {
            'train': read_csv(config, 'train'),
            'valid': read_csv(config, 'valid'),
            'test': read_csv(config, 'test')
        }

    return dataset
  

def prepare_meta(config, dataset):
    logger.log(f"- 1.2 Prepare metadata", level=2)

    desc_path = config.get("DESCRIPT_PATH", None)
    if desc_path is None:
        return None


    meta = create_meta_info(config, dataset)
    meta["desc"] = read_description(config, meta)

    return meta



def read_csv(config, category):
    file_path = os.path.join(config['DATASET_PATH'], f"{category}.csv")
    index_col = config.get('INDEX_LABEL', None)

    try:
        csv_file = pd.read_csv(file_path, index_col=index_col)            
        logger.log(f"- {category:5} data{csv_file.shape} is now loaded", level=3)

        return csv_file
    except FileNotFoundError:
        return None



def read_description(config, meta):
    file_path = os.path.join(config["DATASET_PATH"], config["DESCRIPT_PATH"])

    try:
        with open(file_path, "r", newline="\r\n") as desc_file:
            logger.log(f"- '{config['DESCRIPT_PATH']}' is now loaded", level=3)

            desc_list = desc_file.read().splitlines()
            for desc_line in desc_list:
                col, desc = desc_line.split(":")
                
                col.strip()
                meta[col] = dict()
                meta['DESC'] = desc.strip()
            
        return meta
            

    except FileNotFoundError as e:
        logger.log(f"Description File Not Found Error, '{file_path}' {e}")
        return None
        

def create_meta_info(config, dataset):
    logger.log(f"- Create metadata", level=3)

    train_set, test_set = dataset['train'], dataset['test']
    train_col, test_col = train_set.columns, test_set.columns

    meta = dict()
    meta["__target__"] = config.get("TARGET_LABEL", train_col.difference(test_col).values)
    meta["__nrows__"] = {
        "train": len(train_set),
        "test": len(test_set)
    }
    meta["__ncolumns__"] = len(train_col)
    meta["__columns__"] = train_col.values

    for i, col in enumerate(meta["__columns__"]):
        col_data = train_set[col]

        meta[col] = {
            "index": i,
            "dtype": convert_dict(col_data.dtype),
            "descript": None,
            "nunique": col_data.nunique(),
            "na_count": col_data.isna().sum(),
            "target": (meta["__target__"] == col)
        }

        if meta[col]["dtype"][:3] == "Cat":
            meta[col]["stat"] = {
                "unique": col_data.unique(),
            }
        elif meta[col]["dtype"][:3] == "Num":
            meta[col]["stat"] = {
                "skew": round(col_data.skew(), 4),
                "kurt": round(col_data.kurt(), 4)
            }

    return meta


def convert_dict(dtype):
    return {
        'int64': 'Num_int',
        'float64': 'Num_float',
        'object': 'Cat',
    }[dtype.name]
