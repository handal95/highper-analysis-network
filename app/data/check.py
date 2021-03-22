import structlog
import numpy as np
import pandas as pd

logger = structlog.get_logger()


def check_null_values(config, dataset):
    logger.info(f" - Check Null value data...")

    train_null = get_null_values(dataset['train'], 'train')
    test_null = get_null_values(dataset['test'], 'test')

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
    THRESH = config['CARDINAL_THRESHOLD']

    cat_features = list()
    num_features = list()
    for col in dataset['train'].columns:
        if (dataset['train'][col].dtype == 'object') and (col != config['TARGET_LABEL']):
            cat_features.append(col)
        else:
            num_features.append(col)

    low_card_cols = list()
    mid_card_cols = list()
    high_card_cols = list()

    for col in cat_features:
        col_unique = dataset['train'][col].nunique()
        if col_unique < THRESH[0]:
            low_card_cols.append(col)
        elif col_unique < THRESH[1]:
            mid_card_cols.append(col)
        else:
            high_card_cols.append(col)

    # dataset['train'] = dataset['train'].drop(cat_features, axis=1)
    # dataset['test'] = dataset['test'].drop(cat_features, axis=1)
        
    logger.info(
        f"\n low cardinal cols(<{THRESH[0]}) #{len(low_card_cols)}\n {low_card_cols}"
        f"\n mid cardinal cols(<{THRESH[1]}) #{len(mid_card_cols)}\n {mid_card_cols}"
        f"\nhigh cardinal cols(>={THRESH[1]}) #{len(high_card_cols)}\n {high_card_cols}"
    )

    print(dataset['train'].head(10).T)
    return dataset
