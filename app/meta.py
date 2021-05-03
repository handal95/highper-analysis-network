import pandas as pd


def init_set_info(config, dataset):
    def distribute(target, name):
        try:
            values = dataset[name][target].value_counts()
            length = metaset["__nrows__"][name]
            distribution = round(values / length * 100, 3).to_frame(name=name)
            return distribution
        except:
            # logger.warn(f"{name} set doesn't have {target}", level=3)
            return None
    trainset = dataset["train"]
    testset = dataset["test"]

    train_col = trainset.columns
    test_col = testset.columns

    target_label = config["dataset"].get(
        "target_label", train_col.difference(test_col).values
    )

    metaset = dict()

    metaset["__target__"] = target_label
    metaset["__nrows__"] = {"train": len(trainset), "test": len(testset)}
    metaset["__ncolumns__"] = len(train_col)
    metaset["__columns__"] = pd.Series(train_col.values)
    metaset["__distribution__"] = pd.concat(
        [distribute(target_label, "train"), distribute(target_label, "test")],
        axis=1,
        names=["train", "test"],
    )

    return metaset


def convert_dict(dtype):
    return {
        "Int64": "Num_int",
        "Float64": "Num_float",
        "object": "Cat",
    }[dtype.name]
    

def init_col_info(metaset, col_data, col):
    col_meta = {
        "index": list(metaset["__columns__"]).index("datetime"),
        "name": col,
        "dtype": str(col_data.dtype),
        "descript": None,
        "nunique": col_data.nunique(),
        "na_count": col_data.isna().sum(),
        "target": (metaset["__target__"] == col),
        "log": list(),
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
        
    return col_meta

def updata_set_info(metaset):
    metaset["__target__"] = target
    return metaset

def update_column_info(columns):
    pass
