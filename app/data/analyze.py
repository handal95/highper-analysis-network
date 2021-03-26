import pandas as pd
import numpy as np
from app.utils.logger import Logger
from app.utils.command import request_user_input

logger = Logger()


def analize_dataset(config, dataset):
    logger.log("- 1.1.+ : Check Data Collection", level=2)
    
    logger.log("- 1.1.1 : Check Columns ", level=3)
    train_cols = dataset['train'].columns
    test_cols = dataset['test'].columns

    diff_cols = train_cols.difference(test_cols)
    union_cols = train_cols.union(test_cols)
    
    col_info = []
    pd.set_option('display.max_rows', None)
    dtype_dict = {
        'int64': 'Num.',
        'float64': 'Num.',
        'object': 'Cat.',
    }

    for col in train_cols:
        dtype = dataset['train'][col].dtype.name
        na_count = dataset['train'][col].isna().sum()
        na_percent = np.round((100 * (na_count)/len(dataset['train'])), 2)
        col_unique = dataset['train'][col].nunique()

        info = {
            'Features': col,
            'dType': dtype_dict[dtype],
            'nunique': col_unique,
            'NA (count)': na_count if na_count > 0 else 0,
            'NA (%)': f"{na_percent:4.2f}" if na_percent else " ",
            'Sample1': dataset['train'][col][1],
            'Sample2': dataset['train'][col][2],
            'Sample3': dataset['train'][col][3],
        }
        
        col_info.append(info)

    data_frame = pd.DataFrame(col_info, index=None).sort_values(by='NA (count)', ascending=False)
    description = load_description(config)
    if description is not None:
        data_frame = data_frame.merge(description, how='left', on='Features')
    
    data_frame = data_frame.reindex(
        columns=['Features', 'Description', 'dType', 'Sample1', 'Sample2', 'Sample3', 'nunique', 'NA (count)', 'NA (%)'])

    logger.log(
        f"DATASET Analysis \n"
        f"  Total Train dataset : {len(dataset['train'])}\n"
        f"Inferred Target label : {diff_cols.values} \n"
        f"{data_frame}",
        level=4
    )

    answer = request_user_input(
        "Are there any issues that need to be corrected? ( Y / n )",
        valid_inputs=['Y', 'N'], default='Y'
    )

    return dataset


def load_description(config):
    desc_path = config["DESCRIPT_PATH"]
    if desc_path is None:
        return None

    try:
        with open(desc_path, "r", newline="\r\n") as desc_file:
            desc_list = desc_file.read().splitlines()

            desc_info = []
            for desc_line in desc_list:
                col, desc = desc_line.split(": ")
                
                info = {
                    'Features': col,
                    'Description': desc
                }
                desc_info.append(info)

            
            data_frame = pd.DataFrame(desc_info, index=None)

            return data_frame

    except FileNotFoundError as e:
        logger.info(f"      Description File Not Found Error, '{desc_path}'")
        return None
