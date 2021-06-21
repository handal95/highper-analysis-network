from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)

from model.utils.file import open_csv, open_json
from model.utils.logger import Logger


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

        model_logistic = LogisticRegression()

        models = {
            "Lasso": model_Lasso,
            "ENet": model_ENet,
            "GBoost": model_GBoost,
            "XGBoost": model_XGB,
            "LogReg": model_logistic,
        }

        return models

    def fit_model(self, dataset, metaset):
        dataset["valid"] = dataset["train"][:45569]
        dataset["train"] = dataset["train"][45569:]

        train_label = dataset["train"][metaset["__target__"]]
        train_value = dataset["train"].drop(columns=metaset["__target__"])

        valid_label = dataset["valid"][metaset["__target__"]]
        valid_value = dataset["valid"].drop(columns=metaset["__target__"])

        predicts = dict()
        models = self.models

        def fitting(model, x_train, x_test, y_train, y_test):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            self.metrics(y_test, y_pred)
            return y_pred

        print("FIT - LogReg")
        predicts["LogReg"] = fitting(
            model=models["LogReg"],
            x_train=train_value,
            x_test=valid_value,
            y_train=train_label,
            y_test=valid_label,
        )

        # log_train_predict = (
        #     predicts["Lasso"]
        #     + predicts["ENet"]
        #     + predicts["GBoost"]
        #     + predicts["XGBoost"]
        # ) / 4

        # train_score = mean_squared_error(train_label, log_train_predict)
        # print(f"Scoring with train data : {train_score}")

    def metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_score = roc_auc_score(y_test, y_pred, average="macro")
        print(f"accr : {accuracy:.2f}, prec : {precision:.2f}, recall : {recall:.2f}")
        print(f"f1   : {f1:.2f},  auc : {roc_score:.2f}")
