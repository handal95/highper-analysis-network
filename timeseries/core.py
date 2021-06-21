import argparse

from timeseries.dataset import DataSettings
from timeseries.dataset import NabDataset

def main(opt):
    data_settings = DataSettings(opt.data)

    # define dataset object and data loader object for NAB dataset
    dataset = NabDataset(data_settings=data_settings)

    print(dataset.x.shape, dataset.y.shape)  # check the dataset shape

if __name__ == "__main__":
    # Argument options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config/data_config.yml", help="data.yml path")
    opt = parser.parse_args()

    main(opt)
