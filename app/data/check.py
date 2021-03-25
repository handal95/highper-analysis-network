import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


def check_cardinal_values(config, dataset):
    logger.info(f" - Check Cardinal value data...")
    THRESH = config['CARDINAL_THRESHOLD']

    cat_features = dataset.select_dtypes(include='object').drop(columns=["_SET_"])
    num_features = dataset.select_dtypes(exclude='object')

    card_info_data = []
    for col in cat_features:
        col_unique = dataset[col].nunique()

        category = None
        if col_unique < THRESH[0]:
            category = 'low'
        elif col_unique < THRESH[1]:
            category = 'mid'
        else:
            category = 'high'
        
        card_info = {
            'Features': col,
            'nunique': col_unique,
            'category': category
        }
        card_info_data.append(card_info)

    data_frame = pd.DataFrame(card_info_data, index=None).sort_values(by=f'nunique', ascending=False)
    logger.info(f"\n{data_frame}")

    return dataset


def check_skewness_kurtosis(config, dataset):
    logger.info(f" - Check Numeric value data...")

    num_features = dataset.select_dtypes(exclude='object')
    log_dataset = np.log1p(num_features)
    sqrt_dataset = np.sqrt(num_features)
    log_sqrt_dataset = np.log1p(np.sqrt(num_features))
    
    info_data = []
    for col in num_features:
        info = {
            'Features' : col,

            'Skew': f"{dataset[col].skew():5.2f}",
            'S(L)': f"{log_dataset[col].skew():5.2f}",
            'S(S)': f"{sqrt_dataset[col].skew():5.2f}",
            'S(LS)': f"{log_sqrt_dataset[col].skew():5.2f}",

            'Kurt': f"{dataset[col].kurt():5.2f}",
            'K(L)': f"{log_dataset[col].kurt():5.2f}",
            'K(S)': f"{sqrt_dataset[col].kurt():5.2f}",
            'K(LS)': f"{log_sqrt_dataset[col].kurt():5.2f}",
        }
        info_data.append(info)

    data_frame = pd.DataFrame(info_data, index=None).sort_values(by=f'Features', ascending=False)
    logger.info(f"\n{data_frame}")

    return dataset
