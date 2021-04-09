import argparse
# from app.run import run
from app.utils.logger import Logger, init_logger
from app.data.prepare import prepare_data
from app.loader import DataLoader

argparser = argparse.ArgumentParser(description='cli command')

argparser.add_argument(
    '-t', '--type', help='type of target dataset')

argparser.add_argument(
    '-s', '--skip', help='skipped process')

def _main_(args):
    logger = Logger()

    logger.log("Step 1 >> Data Preparation")
    logger.log("- 1 : Data Collection ", level=1)

    loader = DataLoader(config_path="./config.json")

    loader.analize_dataset()
    # run()


if __name__ == '__main__':
    init_logger()
    _main_(argparser.parse_args())
