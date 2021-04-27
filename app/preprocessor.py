from app.utils.file import open_json


class PreProcessor(object):
    def __init__(self, config_path, dataset, metaset):
        self.config = open_json(config_path)
        self.trainset = dataset["train"]
        self.testset = dataset["test"]
        self.metaset = metaset

    def label_split(self):
        x_label = self.trainset[self.metaset["__target__"]]
        x_value = self.trainset.drop(columns=self.metaset["__target__"])

        y_label = self.testset[self.metaset["__target__"]]
        y_value = self.testset.drop(columns=self.metaset["__target__"])

        return (x_value, x_label), (y_value, y_label)
