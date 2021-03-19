import structlog
import numpy as np
import pandas as pd

logger = structlog.get_logger()


def check_null_values(config, dataset):
    logger.info(f" - Check Null value data...")
    (train_data, valid_data, test_data) = dataset

    train_null = get_null_values(train_data, 'train')
    test_null = get_null_values(test_data, 'test')

    null_display = pd.merge(train_null, test_null, how='outer', on='Features')
    logger.info(f"\n{null_display}")

    return dataset


def get_null_values(dataset, name):
    if dataset is None:
        return None

    na_info_data = []
    for col in dataset.columns:
        na_count = dataset[col].isna().sum()
        if na_count > 0:
            na_percent = np.round((100 * (na_count)/len(dataset)), 2)            
            na_info ={
                'Features' : col,
                f'NA_{name} (count)': na_count,
                f'NA_{name} (%)': na_percent
            }
            na_info_data.append(na_info)

    data_frame = pd.DataFrame(na_info_data, index=None).sort_values(by=f'NA_{name} (count)', ascending=False)

    return data_frame


def check_cardinal_values(config, dataset):
    logger.info(f" - Check Cardinal value data...")
    (train_data, valid_data, test_data) = dataset
    THRESH = config['CARDINAL_THRESHOLD']

    cat_features = list()
    num_features = list()
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            cat_features.append(col)
        else:
            num_features.append(col)

    low_cardinal_cols = list()
    mid_cardinal_cols = list()
    high_cardinal_cols = list()

    for col in cat_features:
        col_unique = train_data[col].nunique()
        if col_unique < THRESH[0]:
            low_cardinal_cols.append(col)
        elif col_unique < THRESH[1]:
            mid_cardinal_cols.append(col)
        else:
            high_cardinal_cols.append(col)
    
    logger.info(
        f"\n low cardinal cols(<{THRESH[0]}) \n {low_cardinal_cols}"
        f"\n mid cardinal cols(<{THRESH[1]}) \n {mid_cardinal_cols}"
        f"\nhigh cardinal cols(>={THRESH[1]}) \n {high_cardinal_cols}"
    )

    return dataset
