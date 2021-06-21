import pandas as pd


def init_set_info(config, dataset):
    def distribute(target, name):
        try:
            values = dataset[name][target].value_counts()
            length = metaset["__nrows__"][name]
            distribution = round(values / length * 100, 3).to_frame(name=name)
            return distribution
        except:
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
        "Int64": "Num_int  ",
        "Float64": "Num_float",
        "object": "Cat     ",
        "string": "string  ",
    }[dtype]


def init_col_info(metaset, col_data, col_name):
    col_meta = {
        "index": list(metaset["__columns__"]).index(col_name),
        "name": col_name,
        "dtype": convert_dict(str(col_data.dtype)),
        "descript": None,
        "nunique": col_data.nunique(),
        "na_count": col_data.isna().sum(),
        "target": (metaset["__target__"] == col_name),
        "log": list(),
    }

    if col_meta["dtype"][:3] == "Cat":
        col_meta["unique"] = (col_data.unique(),)
        col_meta["rate"] = (col_data.value_counts(),)
    elif col_meta["dtype"] == "Int64" or col_meta["dtype"] == "Float64":
        col_meta["stat"] = {
            "skew": round(col_data.skew(), 4),
            "kurt": round(col_data.kurt(), 4),
            "unique": col_data.unique(),
        }

    return col_meta


def add_col_info(metaset, col_data, col_name, descript=None):
    metaset["__ncolumns__"] = metaset["__ncolumns__"] + 1
    metaset["__columns__"] = metaset["__columns__"].append(
        pd.Series(col_name), ignore_index=True
    )

    metaset[col_name] = {
        "index": list(metaset["__columns__"]).index(col_name),
        "name": col_name,
        "dtype": convert_dict(str(col_data.dtype)),
        "descript": descript,
        "nunique": col_data.nunique(),
        "na_count": col_data.isna().sum(),
        "target": False,
        "log": list(),
    }

    return metaset, col_data


def get_meta_info(metaset, dataset):
    info = list()
    for col in metaset["__columns__"]:
        col_meta = metaset[col]
        col_info = {
            "name": col,
            "dtype": col_meta["dtype"],
            "desc": col_meta["descript"],
        }
        for i in range(1, 6):
            col_info[f"sample{i}"] = dataset["train"][col][i]

        info.append(col_info)

    info_df = pd.DataFrame(info)
    return info_df


def show_col_info(col_meta, col_data):
    print(col_meta)
    print(
        f"[{(col_meta['index']):3d}] \n"
        f"<< {col_meta['name']} >> \n"
        f" - {col_meta['descript']}"
    )

    if col_meta["dtype"] == "Datetime":
        print(
            f" === datetime stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" na count : {col_meta['na_count']} \n"
            f" min value: {min(col_data)} \n"
            f" max value: {max(col_data)} \n"
        )
        if col_meta["nunique"] <= 10:
            print()

    elif col_meta["dtype"][:3] == "Num":
        print(
            f" === Numerical stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" na count : {col_meta['na_count']} \n"
        )
        if col_meta.get("stat") is not None:
            print(
                f" skew     : {col_meta['stat'].get('skew', None)} \n"
                f" kurt     : {col_meta['stat'].get('kurt', None)} \n"
                f" values   : {col_meta['stat']['unique'][:10]} ... \n"
            )
        print(col_data.describe(percentiles=[0.03, 0.25, 0.50, 0.75, 0.97]))

    elif col_meta["dtype"] == "Boolean":
        print(
            f" === Boolean stat === \n"
            f" nunique   : {col_meta['nunique']} \n"
            f" na count  : {col_meta['na_count']} \n"
            f" true  rate: {col_meta['rate'][0]} \n"
            f" false rate: {col_meta['rate'][1]} \n"
        )

    elif col_meta["dtype"] == "Category":
        print(
            f" === Categorical stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" unique   : {col_meta['unique']} \n"
            f" na count : {col_meta['na_count']} \n"
        )
        for value in col_meta["rate"]:
            print(f" values   : {value}")

    for log in col_meta["log"]:
        print(f" Log : {log}")

    print()


def update_set_info(metaset):
    metaset["__target__"] = target
    return metaset


def update_col_info(metaset, col_data):
    col_meta = {
        "index": list(metaset["__columns__"]).index(col_name),
        "name": col_name,
        "dtype": convert_dict(str(col_data.dtype)),
        "descript": None,
        "nunique": col_data.nunique(),
        "na_count": col_data.isna().sum(),
        "target": (metaset["__target__"] == col_name),
        "log": list(),
    }
    return col_meta
