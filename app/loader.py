import os
from app.utils.file import open_json, open_csv
from app.utils.logger import Logger
from app.utils.command import request_user_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader(object):
    """
    Data Loader
    """
    def __init__(self, config_path):
        self.config = open_json(config_path)

        self.filepath = self.config["dataset"]["filepath"]
        self.basepath = self.config["dataset"]["dirpath"]
        self.format = self.config["dataset"]["format"].lower()

        self.logger = Logger()
        self.dataset = self.load_dataset()
        self.metaset = self.load_metaset()

    def load_dataset(self):
        self.logger.log(f"- {self.config['dataset']['category']} type dataset loading .. ", level=2)

        #if self.format == "csv" TODO : Fill some other type of data
        filepath = os.path.join(self.basepath, self.filepath + "." + self.format)

        self.logger.log(f"- '{filepath}' is now loading...", level=2)

        dataset = {
            'train': self.read_csv('train'),
            'valid': self.read_csv('valid'),
            'test': self.read_csv('test')
        }

        if dataset['test'] is None:
            dataset = self.split_dataset(dataset, 'train', 'test')

        return dataset

    def read_csv(self, name):
        filepath = os.path.join(self.basepath, self.filepath, name + "." + self.format)
        
        index_col = self.config["dataset"].get('index', None) 
        index_col = index_col if index_col != 'None' else None

        try:
            csv_file = open_csv(filepath=filepath, index_col=index_col)
            self.logger.log(f"- {name:5} data{csv_file.shape} is now loaded", level=3)

        except FileNotFoundError:
            csv_file = None

        return csv_file

    def split_dataset(self, dataset, origin, target):
        split_ratio = self.config['dataset']['split_ratio']

        dataset[origin], dataset[target] = train_test_split(
            dataset[origin], train_size=split_ratio, random_state=42)

        self.logger.log(
            f"- {origin:5} data{dataset[origin].shape}"
            f", {target:5} data{dataset[target].shape}"
            f"  (split ratio: {split_ratio})", level=3)

        return dataset

    
    def load_metaset(self):
        self.logger.log(f"- 1.2 Prepare metadata", level=2)

        def convert_dict(dtype):
            return {
                'int64': 'Num_int',
                'float64': 'Num_float',
                'object': 'Cat',
            }[dtype.name]

        desc_path = self.config.get("DESCRIPT_PATH", None)
        dataset = self.dataset

        metaset = dict()

        trainset, testset = dataset['train'], dataset['test']
        train_col, test_col = trainset.columns, testset.columns

        target_label = self.config["dataset"].get("target_label", train_col.difference(test_col).values)
        
        metaset["__target__"] = target_label
        metaset["__nrows__"] = {"train": len(trainset), "test": len(testset)}
        metaset["__ncolumns__"] = len(train_col)
        metaset["__columns__"] = train_col.values
        train_distribution = dataset['train'][target_label].value_counts()
        test_distribution = dataset['test'][target_label].value_counts()
        metaset["__distribution__"] = {
            'train': (train_distribution/len(trainset))*100,
            'test': (test_distribution/len(testset))*100,
        }

        for i, col in enumerate(metaset["__columns__"]):
            col_data = trainset[col]
            col_meta = {
                "index": i,
                "name": col,
                "dtype": convert_dict(col_data.dtype),
                "descript": None,
                "nunique": col_data.nunique(),
                "na_count": col_data.isna().sum(),
                "target": (metaset["__target__"] == col),
                "log": list()
            }

            if col_meta["dtype"][:3] == "Cat":
                col_meta["stat"] = {
                    "unique": col_data.unique(),
                }
                col_meta["dtype"] = f"{col_meta['dtype']}_{col_meta['nunique']}"
            elif col_meta["dtype"][:3] == "Num":
                col_meta["stat"] = {
                    "skew": round(col_data.skew(), 4),
                    "kurt": round(col_data.kurt(), 4),
                    "unique": col_data.unique(),
                }

            metaset[col] = col_meta

        return metaset


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

        for i, col in enumerate(metaset['__columns__']):
            col_meta = metaset[col]
            self.logger.log(
                f"{col_meta['index']:3d} "
                f"{col_meta['name']:20} "
                f"{col_meta['dtype']:10} "
                f"{col_meta['descript']}"
            )
        
        self.config['options']['FIX_COLUMN_INFO'] = request_user_input(
            "Are there any issues that need to be corrected? ( Y / n )",
            valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
        )

        if self.config['options']['FIX_COLUMN_INFO'] is True:
            self.analize_feature()

    def analize_feature(self):
        self.logger.log("- 1.1.+ : Check Data Features", level=2)


        for i, col in enumerate(self.metaset['__columns__']):
            col_meta = self.metaset[col]
            col_data = self.dataset['train'][col]
            print_meta_info(col_meta, col_data)

            sns.displot(col_data.values, color='r')

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
        plt.show()

        return dataset


def print_meta_info(col_meta, col_data):
    print(
        f"[{(col_meta['index']):3d}] \n"
        f"<< {col_meta['name']} >> \n"
        f" - {col_meta['descript']}"
    )

    if col_meta['dtype'][:3] == "Num":
        print(f" === Numerical stat === \n")
        print(f" skew     : {col_meta['stat']['skew']} ") if col_meta['stat'].get('skew', None) else None
        print(f" kurt     : {col_meta['stat']['kurt']} ") if col_meta['stat'].get('kurt', None) else None
        print(
            f" nunique  : {col_meta['nunique']} \n"
            f" values  : {col_meta['stat']['unique'][:10]} ... \n"
            f" na count : {col_meta['na_count']}"
        )
        print(col_data.describe(percentiles=[.03, .25, .50, .75, .97]))
    else:
        print(
            f" === Categorical stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" values   : {col_meta['stat']['unique']} \n"
            f" na count : {col_meta['na_count']}"
        )

    for log in col_meta['log']:
        print(f" Log : {log}")

    print()