import argparse


def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config/data_config.yml", help="data.yml path")
    opt = parser.parse_args()
    return opt