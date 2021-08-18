import argparse
import yaml


class Args:
    def __init__(self):
        parser = self.get_parser()

        self.config_path = parser.config
        self.opt = parser

    def get_parser(self):
        parser = argparse.ArgumentParser(description="** BandGan CLI **")
        parser.set_defaults(function=None)
        parser.add_argument(
            "-cfg",
            "--config",
            type=str,
            default="config/data_config.yml",
            help="config.yml path",
        )
        return parser.parse_args()

    def get_option(self, train=True):
        option = "train" if train else "test"

        with open(self.opt.config) as f:
            config = yaml.safe_load(f)["args"][option]

        return {
            "workers": int(config["workers"]),
            "batch_size": int(config["batch_size"]),
            "epochs": int(config["epochs"]),
            "lr": config["lr"],
        }

    def __str__(self):
        return str(self.opt)
