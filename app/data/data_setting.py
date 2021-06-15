import yaml

class DataSettings:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)["data"]

        end_name = config["end_name"]

        self.train = config["train"]
        self.window_length = config["window_length"]

        self.BASE = config["BASE"]
        self.label_file = config["label"]
        self.data_file = config["data"] + end_name
        self.key = config["key"] + end_name