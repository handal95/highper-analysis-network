import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

def scale_feature(config, dataset):
    logger.info(f"   - Scaling data")

    Scaler = StandardScaler()

    dataset['train'] = pd.DataFrame(Scaler.fit_transform(dataset['train']))
    dataset['test'] = pd.DataFrame(Scaler.fit_transform(dataset['test']))

    return dataset
