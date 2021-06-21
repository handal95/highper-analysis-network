import pandas as pd
import structlog
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()


def one_hot_encoding(config, dataset):
    logger.info(f"   - One hot encoding")
    Scaler = StandardScaler()
    imputer = SimpleImputer(strategy="mean")

    dataset_label = dataset[config["TARGET_LABEL"]]
    dataset = dataset.drop(columns=config["TARGET_LABEL"])

    category_cols = dataset.select_dtypes(include="object")
    category_nums = dataset.select_dtypes(exclude="object")
    category_dums = pd.get_dummies(category_cols, drop_first=True)
    print(category_nums[:20])

    category_nums = pd.DataFrame(imputer.fit_transform(category_nums))
    print(category_nums[:20])

    category_nums = pd.DataFrame(Scaler.fit_transform(category_nums))
    imputer.fit(category_nums)

    dataset = pd.merge(category_nums, category_dums, left_index=True, right_index=True)
    dataset = pd.concat([dataset, dataset_label], axis=1)

    return dataset
