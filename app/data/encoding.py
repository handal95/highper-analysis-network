import structlog


logger = structlog.get_logger()


def onehot(series):
    logger.info(f" - one-hot encoding ")

    series_new = pd.DataFrame()
    for col in series.unique():
        series_new[name + "_{0}".format(col)] = series == col
    
    return (series_new.astype(int), series_new.columns)