import structlog

logger = structlog.get_logger()


def split_label_feature(config, dataset):
    logger.info(f"   - Splitting target label")

    (train_data, train_label) = split_label(config, dataset["train"], "train")
    (test_data, test_label) = split_label(config, dataset["test"], "test")
    return (train_data, train_label), (test_data, test_label)


def shuffle_train_data(config, dataset):
    if config["DATA_SHUFFLE"] is False:
        return dataset
    logger.info(f"   - Shuffling train data")
    (train_data, train_label), (test_data, test_label) = dataset

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_label = train_label.sample(frac=1).reset_index(drop=True)

    dataset = (train_data, train_label), (test_data, test_label)
    return dataset


def split_label(config, dataset, name):
    if dataset is None:
        return (None, None)

    try:
        label = dataset[config["TARGET_LABEL"]]
        data = dataset.drop(columns=config["TARGET_LABEL"])
        logger.info(
            f"        {name:5} Label shape {label.shape}, Dataset shape {dataset.shape}"
        )
        return (data, label)
    except:
        if name == "test":
            logger.info(
                f"        {name:5} doesn't have [{config['TARGET_LABEL']}] label column"
            )
            return (dataset, None)
        else:
            raise


def split_train_valid(config, dataset):
    logger.info(f"   - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    (train_data, train_label), (test_data, test_label) = dataset

    total_length = len(train_data)
    split_idx = int(total_length * config["SPLIT_RATE"])

    valid_data = train_data[split_idx:]
    valid_label = train_label[split_idx:]

    train_data = train_data[:split_idx]
    train_label = train_label[:split_idx]
    logger.info(
        f"     - total ({total_length}), train ({len(train_label)}) valid ({len(valid_label)})"
    )

    return (train_data, train_label), (valid_data, valid_label)
