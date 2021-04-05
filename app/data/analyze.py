import os
import numpy as np
import pandas as pd
from app.utils.logger import Logger
from app.utils.command import request_user_input

logger = Logger()


def analize_dataset(config, dataset, metaset):
    logger.log("- 1.2 : Analize Data", level=2)
      
    logger.log(
        f"DATASET Analysis \n"
        f"  Total Train dataset : {metaset['__nrows__']['train']}  - "
        f"Inferred Target label : {metaset['__target__']} \n"
    )

    config['options']['FIX_COLUMN_INFO'] = request_user_input(
        "Are there any issues that need to be corrected? ( Y / n )",
        valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
    )

    return dataset


def load_description(config):
    desc_path = config.get("DESCRIPT_PATH")
    if desc_path is None:
        return None

    file_path = os.path.join(config['DATASET_PATH'], desc_path)
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
        logger.log(f"      Description File Not Found Error, '{desc_path}'")
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
