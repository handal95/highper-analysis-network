import pandas as pd

from app.utils.command import request_user_input
from app.utils.file import open_json
from app.utils.logger import Logger
from app.utils.eda import EDA
from app.meta import add_col_info, get_meta_info

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
            # " Label Distribution  :\n{metaset['__distribution__']} \n"
        )

        self.eda.countplot(
            dataframe=dataset["train"],
            column=metaset["__target__"],
            title="Target Label Distributions",
        )

        request_user_input()

        self.analize_dtype()

    def analize_dtype(self):
        self.logger.log(" - 2.1 Analize Dtype", level=2)

        # SHOW INFO
        info_df = get_meta_info(self.metaset, self.dataset)
        print(info_df)

        # USER COMMAND
        answer = request_user_input(
            "Are there any issues that need to be corrected? ( Y / n )",
            valid_inputs=["Y", "N"],
            valid_outputs=[True, False],
            default="Y",
        )
        while answer:
            target_index = int(
                request_user_input(
                    f"Please enter the index of the target to be modified.",
                    valid_inputs=range(self.metaset["__ncolumns__"]),
                    skipable=True,
                    default=None,
                )
            )

            if target_index is None:
                break

            self.logger.log(f"\n{info_df.loc[target_index]}")
            target_col = self.metaset["__columns__"][target_index]

            right_dtype = request_user_input(
                f"Please enter right dtype [num-int, num-float, bool, datetime]",
                valid_inputs=["num-int", "num-float", "bool", "datetime"],
                skipable=True,
                default=None,
            )

            print(f"you select dtype {right_dtype}")
            
            if right_dtype == "Bool":
                self.dataset["train"][target_col] = self.dataset["train"][
                    target_col
                ].replace({True: 1, False: 0}, inplace=True)
                print(self.dataset["train"][target_col][1:5])

            if right_dtype == "Cat":
                self.dataset["train"][target_col].convert_dtypes()

            if right_dtype == "Datetime":
                self.convert_dtype(target_col, right_dtype)

            info_df = get_meta_info(self.metaset, self.dataset)
            print(info_df)
            answer = request_user_input(
                "Are there more issues that need to be corrected? ( Y / n )",
                valid_inputs=["Y", "N"],
                valid_outputs=[True, False],
                default="Y",
            )

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

        self.config["options"]["FIX_COLUMN_INFO"] = request_user_input(
            "Are there any issues that need to be corrected? ( Y / n )",
            valid_inputs=["Y", "N"],
            valid_outputs=[True, False],
            default="Y",
        )

        if self.config["options"]["FIX_COLUMN_INFO"] is True:
            self.analize_feature()

    def analize_feature(self):
        self.logger.log("- 1.1.+ : Check Data Features", level=2)

        for i, col in enumerate(self.metaset["__columns__"]):
            col_meta = self.metaset[col]
            col_data = self.dataset["train"][col]
            show_meta_info(col_meta, col_data)

            # if min(col_data) >= 0:
            #     col_values = col_data.values
            #     log_values = np.log1p(col_values)
            #     sqrt_values = np.sqrt(col_values)

            #     fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            #     sns.histplot(col_values, ax=ax[0], color='r')
            #     ax[0].set_xlim([min(col_values), max(col_values)])

            #     sns.histplot(log_values, ax=ax[1], color='r')
            #     ax[1].set_xlim([min(log_values), max(log_values)])

            #     sns.histplot(sqrt_values, ax=ax[2], color='r')
            #     ax[2].set_xlim([min(sqrt_values), max(sqrt_values)])
            #     plt.show()

            # if col_meta['dtype'][:3] == 'Num':
            #     dataset = convert_scale_domain(config, info, dataset, metaset)

            # # if info['dtype'][:3] == 'Cat':
            # #     dataset = convert_score_domain(info, dataset, metaset)

            # answer = request_user_input()

            # if answer:
            #     change_options = request_user_input(
            #         "how do you want to change this column ( [b:Bool, s:score] )",
            #         valid_inputs=['b', 's'], valid_outputs=['b', 's'], default='N'
            #     )
            #     if change_options == 'b':
            #         convert_boolean_domain(col, dataset)
            #     elif change_options == 's':
            #         convert_score_domain(col, dataset)

        return self.dataset

    def convert_dtype(self, col, right_dtype):
        if right_dtype == "Datetime":
            self.convert_datetime(col)

    def convert_datetime(self, col):
        self.dataset["train"][col] = pd.to_datetime(self.dataset["train"][col])
        self.metaset[col]["log"].append(
            f"dtype changed : {self.metaset[col]['dtype']} to Datetime")
        self.metaset[col]["dtype"] = "Datetime"

        answer = request_user_input(
            "Do you want to split datetime? ( Y / n )",
            valid_inputs=["Y", "N"],
            valid_outputs=[True, False],
            default="Y",
        )

        if answer:
            metaset, trainset = self.metaset, self.dataset["train"]

            metaset, trainset[f"{col}_year"]  = add_col_info(metaset, trainset[col].dt.year, f"{col}_year")
            metaset, trainset[f"{col}_month"] = add_col_info(metaset, trainset[col].dt.month, f"{col}_month")
            metaset, trainset[f"{col}_day"]   = add_col_info(metaset, trainset[col].dt.day, f"{col}_day")
            metaset, trainset[f"{col}_hour"]  = add_col_info(metaset, trainset[col].dt.hour, f"{col}_hour")
            metaset, trainset[f"{col}_dow"]   = add_col_info(metaset, trainset[col].dt.day_name(), f"{col}_dow")

            self.metaset = metaset
            self.dataset["train"] = trainset

    
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

def show_col_info(col_meta, col_data):
    print(
        f"[{(col_meta['index']):3d}] \n"
        f"<< {col_meta['name']} >> \n"
        f" - {col_meta['descript']}"
    )

    if col_meta["dtype"][:3] == "Num":
        print(f" === Numerical stat === \n")
        print(f" skew     : {col_meta['stat']['skew']} ") if col_meta["stat"].get(
            "skew", None
        ) else None
        print(f" kurt     : {col_meta['stat']['kurt']} ") if col_meta["stat"].get(
            "kurt", None
        ) else None
        print(
            f" nunique  : {col_meta['nunique']} \n"
            f" values  : {col_meta['stat']['unique'][:10]} ... \n"
            f" na count : {col_meta['na_count']}"
        )
        print(col_data.describe(percentiles=[0.03, 0.25, 0.50, 0.75, 0.97]))
    else:
        print(
            f" === Categorical stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" values   : {col_meta['stat']['unique']} \n"
            f" na count : {col_meta['na_count']}"
        )

    for log in col_meta["log"]:
        print(f" Log : {log}")

    print()
