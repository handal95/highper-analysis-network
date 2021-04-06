import pandas as pd

def open_csv(filepath, index_col=None):
    try:
        csv_file = pd.read_csv(filepath, index_col=index_col)
        return csv_file
    except FileNotFoundError:
        return None
