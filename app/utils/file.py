import json
import pandas as pd

def open_csv(filepath, index_col=None, forced=False):
    csv_file = pd.read_csv(filepath, index_col=index_col)
    return csv_file

def open_json(filepath, forced=False):
    json_data = json.load(open(filepath))
    return json_data
        