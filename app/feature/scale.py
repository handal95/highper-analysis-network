import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

def scale_feature(config, dataset):
    logger.info(f"   - Scaling data")

    Scaler = StandardScaler()
    (train_data, train_label), (test_data, test_label) = dataset

    train_data = pd.DataFrame(Scaler.fit_transform(train_data))
    test_data = pd.DataFrame(Scaler.fit_transform(test_data))

    return (train_data, train_label), (test_data, test_label)
