import matplotlib.pyplot as plt
import seaborn as sns


class EDA(object):
    def __init__(self, config):
        self.config = config

    def countplot(self, dataframe, column, title):
        if self.config["visualization"] == "False":
            return
        sns.countplot(x=column, data=dataframe)
        plt.title(title)
        plt.show()
