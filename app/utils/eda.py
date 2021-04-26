import matplotlib.pyplot as plt
import seaborn as sns


class EDA(object):
    def __init__(self, config):
        self.config = config

    @classmethod
    def countplot(cls, dataframe, column, title):
        sns.countplot(x=column, data=dataframe)
        plt.title(title)
        plt.show()
