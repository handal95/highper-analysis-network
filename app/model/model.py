import structlog
from catboost import CatBoostRegressor


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

    return model