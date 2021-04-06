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
        f"  Total Train dataset : {metaset['__nrows__']['train']} \n"
        f"  Total Test  dataset : {metaset['__nrows__']['test']} \n"
        f"  Total Columns num   : {metaset['__ncolumns__']}  \n"
        f"Inferred Target label : {metaset['__target__']} \n"
    )

    for i, col in enumerate(metaset['__columns__']):
        info = metaset[col]
        print(
            f"{info['index']:3d} "
            f"{info['name']:20} "
            f"{info['dtype']:10} "
            f"{info['descript']}"
        )

#        dataset = convert_scale_domain(info, dataset, metaset)
#        dataset = convert_score_domain(info, dataset, metaset)

    config['options']['FIX_COLUMN_INFO'] = request_user_input(
        "Are there any issues that need to be corrected? ( Y / n )",
        valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
    )

    return dataset


def analize_feature(config, dataset, metaset):
    logger.log("- 1.1.+ : Check Data Features", level=2)

    if not config['options']['FIX_COLUMN_INFO']:
        logger.log("- Skipped ", level=3)
        return dataset

    for i, col in enumerate(metaset['__columns__']):
        info = metaset[col]

        print_meta_info(info)

        if info['dtype'][:3] == 'Num':
            dataset = convert_scale_domain(config, info, dataset, metaset)

        # if info['dtype'][:3] == 'Cat':
        #     dataset = convert_score_domain(info, dataset, metaset)

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


def convert_scale_domain(config, info, dataset, metaset):
    col = info['name']

    if abs(info['stat']['skew']) > 0:
        raw_set = dataset['train'][col]

        df = pd.DataFrame(data={
            'raw': raw_set,
            'log': np.log1p(raw_set),
            'sqrt': np.sqrt(raw_set),
            'log_sqrt': np.log1p(np.sqrt(raw_set)),
        })

        skewes = {
            'raw': round(df['raw'].skew(), 4),
            'log': round(df['log'].skew(), 4),
            'sqrt': round(df['sqrt'].skew(), 4),
            'log_sqrt': round(df['log_sqrt'].skew(), 4)
        }

        print(df.describe().T)

        auto_answer = 'R'
        if 'square' in info['descript'] and \
           ((skewes['raw'] > skewes['sqrt']) or (skewes['raw'] > skewes['log_sqrt'])):
           print(f"[Recommand] 'Sqrt' or 'Log Sqrt' scaling is recommended for squared values")
           auto_answer = 'log_sqrt'


        answer = request_user_input(
            f"Do you want to normalize value? ( R / l / s / ls ) \n"
            f"Skews info {skewes}",
            valid_inputs=['R', 'L', 'S', 'LS'],
            valid_outputs=['R', 'log', 'sqrt', 'log_sqrt'],
            default='R'
        )  if config['options']['FIX_COLUMN_AUTO'] is False else auto_answer
            
        if answer != 'R':
            dataset['train'][col] = df[answer]
            info["dtype"] = f"Num_scale{answer}"
            info["stat"] = {
                "skew": round(dataset['train'][col].skew(), 4),
                "kurt": round(dataset['train'][col].kurt(), 4),
                "unique": dataset['train'][col].unique(),            
            }
            info["log"].append(f"Scale value to {answer}")
            print_meta_info(info)
        
    return dataset

def convert_score_domain(info, dataset, metaset):
    col = info['name']
    if info['dtype'][:3] != 'Cat':
        return dataset

    scoretype = [
#        {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
#        {'Gd': 5, 'Av': 4, 'Mn': 3, 'No': 2, 'NA': 0},
#        {'Y': 1, 'N': 0}
        #{'GLQ': 5, 'ALQ': 4, 'Rec':3, 'BLQ': 2, 'LwQ': 1, 'Unf': 0, 'NA': -1},
    ]

    match_index = None
    match_value = 0.2
    for i, stype in enumerate(scoretype):
        interset = intersect(info['stat']['unique'], stype)
        unionset = union(info['stat']['unique'], stype)
        if len(interset) / len(unionset) > match_value:
            match_index = i
            match_value = len(interset) / len(unionset)


    print(f"     raw {info['stat']['unique']}")
    if match_index is not None:
        # answer = request_user_input(
        #     f"[Recommand] Convert Cat value to Score ( Y / n ) \n Mapping {scoretype[match_index]}",
        #     valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
        # )
        # if answer is True:
        if True:
            dataset['train'][col] = dataset['train'][col].replace(scoretype[match_index])

            info["dtype"] = "Num_score"
            info["stat"]["unique"] = dataset['train'][col].unique()
            info["log"].append(
                f"Convert value {scoretype[match_index]}"
                f"interset {intersect(info['stat']['unique'], scoretype[match_index])}"
                f"unionset {union(info['stat']['unique'], scoretype[match_index])}"
            )
        
    return dataset



def intersect(a, b):
    return list(set(a) & set(b))

def union(a, b):
    return list(set(a) | set(b))

def print_meta_info(col_meta):
    print(
        f"[{(col_meta['index']):3d}] \n"
        f"<< {col_meta['name']} >> \n"
        f" - {col_meta['descript']}"
    )

    if col_meta['dtype'][:3] == "Num":
        print(f" === Numerical stat === \n")
        print(f" skew     : {col_meta['stat']['skew']} ") if col_meta['stat'].get('skew', None) else None
        print(f" kurt     : {col_meta['stat']['kurt']} ") if col_meta['stat'].get('kurt', None) else None
        print(
            f" nunique  : {col_meta['nunique']} \n"
            f" values  : {col_meta['stat']['unique'][:10]} ... \n"
            f" na count : {col_meta['na_count']}"
        )
    else:
        print(
            f" === Categorical stat === \n"
            f" nunique  : {col_meta['nunique']} \n"
            f" values   : {col_meta['stat']['unique']} \n"
            f" na count : {col_meta['na_count']}"
        )

    for log in col_meta['log']:
        print(f" Log : {log}")

    print()


