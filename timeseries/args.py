import argparse
import yaml


class Args:
    def __init__(self):
        parser = self.get_parser()
        self.opt = parser.parse_args()

    def get_parser(self):
        parser = argparse.ArgumentParser(
            add_help=False, description="Command Line Interface"
        )
        parser.set_defaults(function=None)
        parser.add_argument(
            "--data", type=str, default="config/data_config.yml", help="data.yml path"
        )

        return parser

    def get_option(self, train=True):
        option = "train" if train else "test"

        with open(self.opt.data) as f:
            config = yaml.safe_load(f)["args"][option]

        return {
            "workers": int(config["workers"]),
            "batch_size": int(config["batch_size"]),
            "epochs": int(config["epochs"]),
            "lr": config["lr"],
        }
