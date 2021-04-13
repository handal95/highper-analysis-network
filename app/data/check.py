import numpy as np
import pandas as pd
from app.utils.logger import Logger

logger = Logger()


def check_data(config, dataset):
    logger.log(f" - 1.1.+ : Check Data Collection")

    logger.log(f"   - 1.1.1 : Check Columns ")
    train_cols = dataset["train"].columns
    test_cols = dataset["train"].columns

    #    union_cols = train_cols.union(test_cols)
    #    logger.log(f"   - Total columns ({len(union_cols)}) : \n List \n {union_cols.values} ")

    col_info = []
    pd.set_option("display.max_rows", None)
    dtype_dict = {"int64": "Num_desc", "float64": "Num_cont", "object": "Cat     "}

    #    pd.set_option('display.max_seq_items', None)

    for col in train_cols:
        dtype = dataset["train"][col].dtype.name
        na_count = dataset["train"][col].isna().sum()
        na_percent = (
            np.round((100 * (na_count) / len(dataset["train"])), 2)
            if na_count > 0
            else None
        )

        info = {
            "Features": col,
            "dType": dtype_dict[dtype],
            "NA (count)": na_count,
            "NA (%)": na_percent,
        }
        col_info.append(info)

    data_frame = pd.DataFrame(col_info, index=None).sort_values(
        by=f"NA (%)", ascending=False
    )
    logger.log(f"\n{data_frame}")

    #    describe = dataset['train'].describe(percentiles=[.03, .25, .50, .75, .97]).T
    #    logger.log(f" - DATA describe, len: {len(dataset['train'])} \n{describe}")

    # logger.log(f"   - Inferred the target column")
    # infered_target = dataset['train'].columns.difference(dataset['test'].columns).values
    # logger.log(f" Infered : {infered_target} ( Y / n )")
    # answer = str(input()).capitalize()

    # if answer is None or answer == 'Y':
    #     logger.log("GOOD")
    # else:
    #     logger.log("WHAT IS ACTUAL TARGET VALUE ? ")

    return dataset


def check_cardinal_values(config, dataset):
    THRESH = config["CARDINAL_THRESHOLD"]
    cat_features = dataset.select_dtypes(include="object")
    num_features = dataset.select_dtypes(exclude="object")
    logger.log(f" - Check Cardinal value data... # {cat_features.columns}")

    card_info_data = []
    for col in cat_features:
        col_unique = dataset[col].nunique()

        category = None
        if col_unique < THRESH[0]:
            category = "low"
        elif col_unique < THRESH[1]:
            category = "mid"
        else:
            category = "high"

        card_info = {"Features": col, "nunique": col_unique, "category": category}
        card_info_data.append(card_info)

    data_frame = pd.DataFrame(card_info_data, index=None).sort_values(
        by=f"nunique", ascending=False
    )
    logger.log(f"\n{data_frame}")

    return dataset


def check_skewness_kurtosis(config, dataset):
    logger.log(f" - Check Numeric value data...")

    num_features = dataset.select_dtypes(exclude="object")
    log_dataset = np.log1p(num_features)
    sqrt_dataset = np.sqrt(num_features)
    log_sqrt_dataset = np.log1p(np.sqrt(num_features))

    info_data = []
    for col in num_features:
        info = {
            "Features": col,
            "Skew": f"{dataset[col].skew():5.2f}",
            "S(L)": f"{log_dataset[col].skew():5.2f}",
            "S(S)": f"{sqrt_dataset[col].skew():5.2f}",
            "S(LS)": f"{log_sqrt_dataset[col].skew():5.2f}",
            "Kurt": f"{dataset[col].kurt():5.2f}",
            "K(L)": f"{log_dataset[col].kurt():5.2f}",
            "K(S)": f"{sqrt_dataset[col].kurt():5.2f}",
            "K(LS)": f"{log_sqrt_dataset[col].kurt():5.2f}",
        }
        info_data.append(info)

    data_frame = pd.DataFrame(info_data, index=None).sort_values(
        by=f"Features", ascending=False
    )
    logger.log(f"\n{data_frame}")

    return dataset
