import structlog
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()

def select_feature(config, dataset):
    logger.info(f"   - Select Feature")

    dataset = drop_toomany_null(config, dataset)
    dataset = drop_uncorr_feature(config, dataset)

    return dataset

def show_corr_image(dataset):
    new_corr = dataset.corr()
    heatmap = sns.heatmap(new_corr, linewidths=.5, vmin=0, vmax=1, square=True, cmap="RdYlGn")
    plt.show()


def drop_uncorr_feature(config, dataset):
    logger.info(f"     - Drop uncorr Feature")
    
    corr = dataset.corr()
    drop_cols = corr.index[corr[config["TARGET_LABEL"][0]] < 0.3]
    logger.info(f"     - Dropped cols # {len(drop_cols)} \n {drop_cols}")

    dataset = dataset.drop(drop_cols, axis=1)
    
    return dataset


def drop_toomany_null(config, dataset):
    logger.info(f"     - Drop many null Feature")

    check_null = dataset.isna().sum() / len(dataset)
    drop_cols = check_null[check_null > 0.5].keys()
    logger.info(f"     - Dropped cols # {len(drop_cols)} \n {drop_cols}")

    dataset = dataset.drop(drop_cols, axis=1)

    return dataset
