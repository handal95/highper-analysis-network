import os
from app.utils.file import open_json, open_csv
from app.utils.logger import Logger


class DataLoader(object):
    """
    Data Loader
    """
    def __init__(self, config_path):
        self.config = open_json(config_path)

        self.filepath = self.config["dataset"]["filepath"]
        self.basepath = self.config["dataset"]["dirpath"]
        self.format = self.config["dataset"]["type"].lower()

        self.logger = Logger()
        self.dataset = self.load_dataset()

    def load_dataset(self):

        data_conf = self.config["dataset"]

        if self.format == "csv":
            filepath = os.path.join(
                self.basepath, self.filepath + "." + self.format)
        else:
            # TODO : Fill some other type of data
            filepath = None

        self.logger.log(f"- '{filepath}' is now loading...", level=2)

        dataset = {
            'train': self.read_csv('train'),
            'valid': self.read_csv('valid'),
            'test': self.read_csv('test')
        }

        return dataset

    def read_csv(self, name):
        filepath = os.path.join(self.basepath, self.filepath, name + "." + self.format)

        try:
            csv_file = open_csv(
                filepath=filepath,
                index_col=self.config["dataset"].get('index', None)
            )
            self.logger.log(f"- {name:5} data{csv_file.shape} is now loaded", level=3)
        except FileNotFoundError:
            csv_file = None

        return csv_file            