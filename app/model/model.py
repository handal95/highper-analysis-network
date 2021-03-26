import structlog
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

logger = structlog.get_logger()


def build_model(config):
    logger.info(f" - Building a model [ {config['MODEL']} ]")

    model_Lasso = make_pipeline(
        RobustScaler(), Lasso(alpha =0.000327, random_state=18))

    model_ENet = make_pipeline(
        RobustScaler(), ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18))

    model_GBoost = GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, 
        min_samples_split=10, loss='huber', random_state=18)

    model_XGB=XGBRegressor(
        colsample_bylevel=0.9229733609038979,colsample_bynode=0.21481791874780318,colsample_bytree=0.607964318297635, 
        gamma=0.8989889254961725, learning_rate=0.009192310189734834, max_depth=3, n_estimators=3602, 
        reg_alpha=3.185674564163364e-12,reg_lambda=4.95553539265423e-13, seed=18, subsample=0.8381904293270576,
        verbosity=0
    )

    models = {
        "Lasso" : model_Lasso,
        "ENet" : model_ENet,
        "GBoost" : model_GBoost,
        "XGBoost" : model_XGB
    }

    return models

def fit_model(models, train, valid):
    (train_data, train_label) = train
    (valid_data, valid_label) = valid

    predicts = dict()
    models["Lasso"].fit(train_data, train_label)
    models["ENet"].fit(train_data, train_label)
    models["GBoost"].fit(train_data, train_label)
    models["XGBoost"].fit(train_data, train_label)

    predicts["Lasso"] = models["Lasso"].predict(train_data)
    predicts["ENet"] = models["ENet"].predict(train_data)
    predicts["GBoost"] = models["GBoost"].predict(train_data)
    predicts["XGBoost"] = models["XGBoost"].predict(train_data)

    log_train_predict = (
        predicts["Lasso"] + predicts["ENet"] + predicts["GBoost"] + predicts["XGBoost"]
    )/4
    
    train_score = np.sqrt(mean_squared_error(train_label, log_train_predict))
    print(f"Scoring with train data : {train_score}")
    
    return models

def estimate_model(models):
    return
    print("Best RMSE :", np.sqrt(np.abs(model.best_score_)))
    print("Best RMSLE :", np.log(np.sqrt(np.abs(model.best_score_))))
    print("BestModel  :", model.best_estimator_)
