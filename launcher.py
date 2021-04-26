import argparse

# from app.run import run
from app.utils.logger import Logger, init_logger
from app.data.prepare import prepare_data
from app.loader import DataLoader
from app.analyzer import Dataanalyzer
from app.model import ModelGenerator

argparser = argparse.ArgumentParser(description="cli command")

argparser.add_argument("-t", "--type", help="type of target dataset")

argparser.add_argument("-s", "--skip", help="skipped process")


def _main_(args):
    init_logger()
    logger = Logger()

    logger.log("Step 0 >> Setting ")

    logger.log("Step 1 >> Data Preparation")
    logger.log("- 1 : Data Collection ", level=1)
    loader = DataLoader(config_path="./config.json")

    logger.log("- 2 : Data Analization ", level=1)
    analyzer = Dataanalyzer(
        config_path="./config.json", dataset=loader.dataset, metaset=loader.metaset
    )
    analyzer.analize()

    logger.log("Step 3 >> Model Generation")
    model_generator = ModelGenerator(config_path="./config.json")
    models = model_generator.models

    logger.log("Step 5 >> Model Evaluation")
    models = model_generator.fit_model(
        dataset=analyzer.dataset, metaset=analyzer.metaset
    )

    # run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    _main_(opt)
