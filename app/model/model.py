import structlog
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

logger = structlog.get_logger()


def build_model(config):
    logger.info(f" - Building a model [ {config['MODEL']} ]")

    model = None
    if config['MODEL'] == 'CatBoost':
        model = CatBoostRegressor(
            random_state = 131,
            n_estimators = 1000,
            depth = 8,
            cat_features = cat_features_encoded,
            loss_function = "MAE"
        )
    elif config['MODEL'] == 'XGBoost':
        model = XGBRegressor(
            learning_rate=0.1,
            n_estimators=100,
            reg_alpha=0.001,
            reg_lambda=0.000001,
            n_jobs=-1,
            min_child_weight=3
        )

    return model
