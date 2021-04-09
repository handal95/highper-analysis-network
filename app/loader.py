import os
from app.utils.file import open_json, open_csv
from app.utils.logger import Logger
from sklearn.model_selection import train_test_split

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

        desc_path = self.config.get("DESCRIPT_PATH", None)
        dataset = self.dataset

        metaset = dict()

        train_set, test_set = dataset['train'], dataset['test']
        train_col, test_col = train_set.columns, test_set.columns

        metaset["__target__"] = self.config.get("target_label", train_col.difference(test_col).values)
        metaset["__nrows__"] = {
            "train": len(train_set),
            "test": len(test_set)
        }
        metaset["__ncolumns__"] = len(train_col)
        metaset["__columns__"] = train_col.values

        return metaset
        # meta = create_meta_info(dataset)
        # meta = read_description(meta)       


    def analize_dataset(self):
        self.logger.log(
            f"DATASET Analysis \n"
            f"  Total Train dataset : {metaset['__nrows__']['train']} \n"
            f"  Total Test  dataset : {metaset['__nrows__']['test']} \n"
            f"  Total Columns num   : {metaset['__ncolumns__']}  \n"
            f"Inferred Target label : {metaset['__target__']} \n"
        )

        