from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from app.utils.file import open_csv, open_json
from app.utils.logger import Logger


class ModelGenerator(object):
    def __init__(self, config_path):
        self.config = open_json(config_path)
        self.logger = Logger()

        self.models = self.build_model()

    def build_model(self):
        self.logger.log(f" - Building a model [ {self.config['model']['MODEL']} ]")

        model_Lasso = make_pipeline(
            RobustScaler(), Lasso(alpha=0.000327, random_state=18)
        )

        model_ENet = make_pipeline(
            RobustScaler(), ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18)
        )

        model_GBoost = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="huber",
            random_state=18,
        )

        model_XGB = XGBRegressor(
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
            verbosity=0,
        )

        models = {
            "Lasso": model_Lasso,
            "ENet": model_ENet,
            "GBoost": model_GBoost,
            "XGBoost": model_XGB,
        }

        return models

    def fit_model(self, dataset, metaset):
        train_label = dataset["train"][metaset["__target__"]]
        train_data = dataset["train"].drop(columns=metaset["__target__"])

        predicts = dict()
        models = self.models
        print("FIT - LASSO")
        models["Lasso"].fit(train_data, train_label)
        print("FIT - ENET")
        models["ENet"].fit(train_data, train_label)
        print("FIT - GBOOST")
        models["GBoost"].fit(train_data, train_label)
        print("FIT - XGBOOST")
        models["XGBoost"].fit(train_data, train_label)

        print("PREDICT - LASSO")
        predicts["Lasso"] = models["Lasso"].predict(train_data)
        print("PREDICT - ENET")
        predicts["ENet"] = models["ENet"].predict(train_data)
        print("PREDICT - GBOOST")
        predicts["GBoost"] = models["GBoost"].predict(train_data)
        print("PREDICT - XGBOOST")
        predicts["XGBoost"] = models["XGBoost"].predict(train_data)

        log_train_predict = (
            predicts["Lasso"]
            + predicts["ENet"]
            + predicts["GBoost"]
            + predicts["XGBoost"]
        ) / 4

        train_score = mean_squared_error(train_label, log_train_predict)
        print(f"Scoring with train data : {train_score}")
