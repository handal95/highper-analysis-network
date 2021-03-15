import structlog


logger = structlog.get_logger()


def clean_data(config, dataset):
    logger.info(f" - Cleaning dataset [ {config['DATASET']} ]")
    (train_data, valid_data, test_data) = dataset
    
    DATA_LEN  = len(train_data)
    THRESHOLD_RATE = 0.1
    for col in train_data.columns:
        if train_data[col].dtype != 'object':
            continue

        unique_count = len(train_data[col].unique())
        logger.info(f"Norminal column [{col}] {unique_count}/{DATA_LEN}")
        # if len(train_data[col].unique()) >= threshold:
        #     train_data = train_data.drop(columns=col)
        #     test_data = test_data.drop(columns=col)
        #     if valid_data:
        #         valid_data = valid_data.drop(columns=col)
        # else:
        #     logger.info(f"Norminal column [{col}] is not dropped"
        #                 f"{unique_count}/{DATA_LEN}")
            

    print(train_data.info())

    return (train_data, valid_data, test_data)
