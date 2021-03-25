import structlog
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

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
        xgb_model = XGBRegressor(
            colsample_bylevel=0.9229733609038979,
            colsample_bynode=0.21481791874780318,
            colsample_bytree=0.607964318297635, 
            gamma=0.8989889254961725,
            learning_rate=0.009192310189734834,
            max_depth=3,
            n_estimators=3602, 
            reg_alpha=3.185674564163364e-12,
            reg_lambda=4.95553539265423e-13,
            seed=18,
            subsample=0.8381904293270576,
#            tree_method='gpu_hist',
            verbosity=1
        )

        gbm_param_grid = {
            'subsample': np.arange(0.05, 1, 0.05),
            'max_depth': np.arange(3, 20, 1),
            'colsample_bytree': np.arange(0.1, 1.05, .05)
        }

        model = RandomizedSearchCV(
            estimator=xgb_model,
            n_iter=10,
            scoring='neg_mean_squared_error',
            verbose=2,
            param_distributions=gbm_param_grid,
            cv=4
        )

    return model

def fit_model(model, train):
    (train_data, train_label) = train
    model.fit(train_data, train_label)
    return model

def estimate_model(model):
    print("Best RMSE :", np.sqrt(np.abs(model.best_score_)))
    print("Best RMSLE :", np.log(np.sqrt(np.abs(model.best_score_))))
    print("BestModel  :", model.best_estimator_)
