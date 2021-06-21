import pandas as pd

from model.utils.command import request_user_input
from model.utils.file import open_json
from model.utils.logger import Logger
from model.utils.eda import EDA
from model.meta import add_col_info, get_meta_info, show_col_info, update_col_info


class DataAnalyzer(object):
    def __init__(self, config_path, dataset, metaset):
        self.config = open_json(config_path)

        self.logger = Logger()
        self.eda = EDA(self.config["analyzer"])
        self.dataset = dataset
        self.metaset = metaset

    def analize(self):
        dataset = self.dataset
        metaset = self.metaset

        pd.set_option("display.max_columns", metaset["__ncolumns__"])
        pd.set_option("display.width", 1000)

        self.logger.log(
            f"DATASET Analysis \n"
            f" Total Train dataset : {metaset['__nrows__']['train']} \n"
            f" Total Test  dataset : {metaset['__nrows__']['test']} \n"
            f" Total Columns num   : {metaset['__ncolumns__']}  \n"
            f" Target label        : {metaset['__target__']} \n"
            f" Target dtype        : {dataset['train'][metaset['__target__']].dtype} \n"
        )

        self.eda.countplot(
            dataframe=dataset["train"],
            column=metaset["__target__"],
            title="Target Label Distributions",
        )

        request_user_input()

        self.analize_dtype()  # 2.1
        self.analize_feature()

    def analize_dtype(self):
        self.logger.log(" - 2.1 Analize Dtype", level=2)

        # SHOW INFO
        print(get_meta_info(self.metaset, self.dataset))

        # USER COMMAND
        answer = ask_boolean("Are there any issues that need to be corrected?")

        while answer:
            target_index = request_user_input(
                f"Please enter the index of the target to be modified.",
                valid_inputs=range(self.metaset["__ncolumns__"]),
                skipable=True,
                default=None,
            )

            if target_index is None:
                break

            target_col = self.metaset["__columns__"][int(target_index)]
            self.convert_dtype(target_col)

            print(get_meta_info(self.metaset, self.dataset))
            answer = ask_boolean("Are there any issues that need to be corrected?")

    def analize_dataset(self):
        metaset = self.metaset
        dataset = self.dataset

        self.logger.log(
            f"DATASET Analysis \n"
            f" Total Train dataset : {metaset['__nrows__']['train']} \n"
            f" Total Test  dataset : {metaset['__nrows__']['test']} \n"
            f" Total Columns num   : {metaset['__ncolumns__']}  \n"
            f" Target label        : {metaset['__target__']} \n"
            f"  [train distribute(percent.)]\n{metaset['__distribution__']['train']} \n"
            f"  [test  distribute(percent.)]\n{metaset['__distribution__']['test']} \n"
        )

        request_user_input()

        for i, col in enumerate(metaset["__columns__"]):
            col_meta = metaset[col]
            self.logger.log(
                f"{col_meta['index']:3d} "
                f"{col_meta['name']:20} "
                f"{col_meta['dtype']:10} "
                f"{col_meta['descript']}"
            )

        answer = ask_boolean("Are there any issues that need to be corrected?")
        self.config["options"]["FIX_COLUMN_INFO"] = answer

        if self.config["options"]["FIX_COLUMN_INFO"] is True:
            self.analize_feature()

    def analize_feature(self):
        self.logger.log("- 2.2 : Check Data Features", level=2)

        for i, col in enumerate(self.metaset["__columns__"]):
            col_meta = self.metaset[col]
            col_data = self.dataset["train"][col]
            show_col_info(col_meta, col_data)
            answer = ask_boolean(
                "Are there any issues that need to be corrected?", default="N"
            )

            while answer:
                target = request_user_input(
                    f"Please enter issue [none, dtype]",
                    valid_inputs=["dtype"],
                    skipable=True,
                    default=None,
                )

                if target == "Dtype":
                    self.convert_dtype(col)
                show_col_info(col_meta, col_data)
                answer = ask_boolean(
                    "Are there any issues that need to be corrected?", default="N"
                )

        print(get_meta_info(self.metaset, self.dataset))

        return self.dataset

    def convert_dtype(self, col):
        right_dtype = request_user_input(
            f"Please enter right dtype [num-int, num-float, bool, category, datetime]",
            valid_inputs=["num-int", "num-float", "bool", "category", "datetime"],
            skipable=True,
            default=None,
        )

        print(f"you select dtype {right_dtype}")

        if right_dtype == "Datetime":
            self.convert_datetime(col)
        elif right_dtype == "Category":
            self.convert_category(col)
        elif right_dtype == "Bool":
            self.convert_boolean(col)

    def convert_datetime(self, col):
        self.dataset["train"][col] = pd.to_datetime(self.dataset["train"][col])
        self.metaset[col]["log"].append(
            f"dtype changed : {self.metaset[col]['dtype']} to Datetime"
        )
        self.metaset[col]["dtype"] = "Datetime"

        answer = ask_boolean("Do you want to split datetime?")
        if answer:
            metaset, trainset = self.metaset, self.dataset["train"]

            metaset, trainset[f"{col}_year"] = add_col_info(
                metaset, trainset[col].dt.year, f"{col}_year"
            )
            metaset, trainset[f"{col}_month"] = add_col_info(
                metaset, trainset[col].dt.month, f"{col}_month"
            )
            metaset, trainset[f"{col}_day"] = add_col_info(
                metaset, trainset[col].dt.day, f"{col}_day"
            )
            metaset, trainset[f"{col}_hour"] = add_col_info(
                metaset, trainset[col].dt.hour, f"{col}_hour"
            )
            metaset, trainset[f"{col}_dow"] = add_col_info(
                metaset, trainset[col].dt.day_name(), f"{col}_dow"
            )

            self.metaset = metaset
            self.dataset["train"] = trainset

    def convert_category(self, col):
        col_meta = self.metaset[col]
        col_data = self.dataset["train"][col]

        col_data = col_data.apply(str)
        col_meta["log"].append(f"dtype changed : {col_meta['dtype']} to Category")
        col_meta["dtype"] = "Category"

        col_meta["unique"] = col_data.unique()
        col_meta["rate"] = (col_data.value_counts(),)

        self.metaset[col] = col_meta
        self.dataset["train"][col] = col_data

    def convert_boolean(self, col):
        col_meta = self.metaset[col]
        col_data = self.dataset["train"][col]

        col_data = col_data.apply(str)
        col_meta["log"].append(f"dtype changed : {col_meta['dtype']} to Boolean")
        col_meta["dtype"] = "Boolean"

        col_meta["rate"] = col_data.value_counts()

        self.metaset[col] = col_meta
        self.dataset["train"][col] = col_data

    def get_meta_info(self, columns):
        info = list()
        for col in columns:
            col_meta = self.metaset[col]
            col_info = {
                "name": col,
                "dtype": col_meta["dtype"],
                "desc": col_meta["descript"],
            }
            for i in range(1, 6):
                col_info[f"sample{i}"] = self.dataset["train"][col][i]

            info.append(col_info)
        info_df = pd.DataFrame(info)
        self.logger.log(f" - Dtype \n {info_df}\n\n", level=3)
        return info_df


def ask_boolean(message, default="Y"):
    message += "( Y / n )" if default == "Y" else "( y / N )"

    return request_user_input(
        message,
        valid_inputs=["Y", "N"],
        valid_outputs=[True, False],
        default=default,
    )
