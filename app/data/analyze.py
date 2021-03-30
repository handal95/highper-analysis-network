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
            'col': col,
            'dType': dtype_dict[dtype],
            'nunique': col_unique,
            'NA (count)': na_count if na_count > 0 else 0,
            'NA (%)': f"{na_percent:4.2f}" if na_percent else " ",
            'Sample1': dataset['train'][col][1],
            'Sample2': dataset['train'][col][2],
            'Sample3': dataset['train'][col][3],
        }
        
        col_info.append(info)

    data_frame = pd.DataFrame(col_info, index=None)
    #.sort_values(by='NA (count)', ascending=False)
    description = load_description(config)
    if description is not None:
        data_frame = data_frame.merge(description, how='left', on='col')
    
    config['info'] = data_frame.reindex(
        columns=['col', 'Description', 'dType', 'Sample1', 'Sample2', 'Sample3', 'nunique', 'NA (count)', 'NA (%)'])
    
    logger.log(
        f"DATASET Analysis \n"
        f"  Total Train dataset : {len(dataset['train'])}  - "
        f"Inferred Target label : {diff_cols.values} \n"
        f"{config['info']}",
        level=4
    )

    config['options']['FIX_COLUMN_INFO'] = request_user_input(
        "Are there any issues that need to be corrected? ( Y / n )",
        valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
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
                    'col': col,
                    'Description': desc
                }
                desc_info.append(info)

            
            data_frame = pd.DataFrame(desc_info, index=None)

            return data_frame

    except FileNotFoundError as e:
        logger.info(f"      Description File Not Found Error, '{desc_path}'")
        return None


def analize_feature(config, dataset):
    logger.log("- 1.1.+ : Check Data Features", level=2)

    if not config['options']['FIX_COLUMN_INFO']:
        logger.log("- Skipped ", level=3)
        return dataset

    info = config['info']
    columns = dataset['train'].columns
    for i, col in enumerate(columns):
        print(
            f"[{(i + 1):3d} / {len(columns)}]    "
            f"{col.ljust(15)} - {info['Description'][i]}"
        )
        dtype_info = dtype_analizer(col, info['dType'][i], dataset)

        if info['dType'][i] == 'Cat.':
            print(
                f"              "
                f"{info['dType'][i].ljust(15)} "
                f"- na_count ({dtype_info['na_count']})"
                f", nunique  ({dtype_info['nunique']})"
                f", unique  ({dtype_info['unique']})"
            )
        else:
            print(
                f"              "
                f"{info['dType'][i].ljust(15)} "
            )
        print("\n")
           
        answer = request_user_input(
            "Are there any issues that need to be corrected? ( y / N )",
            valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='N'
        )

        if answer:
            change_options = request_user_input(
                "how do you want to change this column ( [b:Bool, s:score] )",
                valid_inputs=['b', 's'], valid_outputs=['b', 's'], default='N'
            )
            if change_options == 'b':
                convert_boolean_domain(col, dataset)
            elif change_options == 's':
                convert_score_domain(col, dataset)





    # info = config['info']
    # config['meta']['raw_columns'] = length = len(info)

    # for i in range(config['meta']['raw_columns']):
    #     feature = info['col'][i]

    #     samples = list()
    #     for i in range(1, 10):
    #         samples.append(dataset['train'][feature][i])

    #     print(
    #         f"({i:3d}/{length})"
    #         f"    Column    [{feature.ljust(15)}]"
    #         f":  {info[i].Description} \n"

    #         f"    Data Type {info[i].dType.ljust(15)} "
    #         f":   {samples} \n"
    #     )
    #     input()

        


    return dataset


def dtype_analizer(col, dtype, dataset):
    info = dict()
    if dtype == 'Cat.':
        info['na_count'] = dataset['train'][col].isna().sum()
        info['nunique'] = dataset['train'][col].nunique()
        info['unique'] = dataset['train'][col].unique()
    else:
        info['none'] = "bam"

    return info


def convert_boolean_domain(col, dataset):
    logger.log(f" - Convert column({col}) to boolean")


def convert_score_domain(col, dataset):
    logger.log(f" - Convert column({col}) to score")
