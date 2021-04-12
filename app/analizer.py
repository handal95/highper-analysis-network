from app.utils.logger import Logger
from app.utils.file import open_json

class DataAnalizer(object):
    def __init__(self, config_path, dataset=None, metaset=None):
        self.config = open_json(config_path)

        self.logger = Logger()
        self.dataset = dataset
        self.metaset = metaset

    def analize_dataset(self):
        metaset = self.metaset
        dataset = self.dataset

        self.logger.log(
            f"DATASET Analysis \n"
            f" Total Train dataset : {metaset['__nrows__']['train']} \n"
            f" Total Test  dataset : {metaset['__nrows__']['test']} \n"
            f" Total Columns num   : {metaset['__ncolumns__']}  \n"
            f" Target label        : {metaset['__target__']} \n"
            f"  [train distribute(percent.)]\n{metaset['__distribution__']['train']} \n"
            f"  [test  distribute(percent.)]\n{metaset['__distribution__']['test']} \n"
        )

 
        request_user_input()

        for i, col in enumerate(metaset['__columns__']):
            col_meta = metaset[col]
            self.logger.log(
                f"{col_meta['index']:3d} "
                f"{col_meta['name']:20} "
                f"{col_meta['dtype']:10} "
                f"{col_meta['descript']}"
            )
        
        self.config['options']['FIX_COLUMN_INFO'] = request_user_input(
            "Are there any issues that need to be corrected? ( Y / n )",
            valid_inputs=['Y', 'N'], valid_outputs=[True, False], default='Y'
        )

        if self.config['options']['FIX_COLUMN_INFO'] is True:
            self.analize_feature()

    def analize_feature(self):
        self.logger.log("- 1.1.+ : Check Data Features", level=2)

        for i, col in enumerate(self.metaset['__columns__']):
            col_meta = self.metaset[col]
            col_data = self.dataset['train'][col]
            print_meta_info(col_meta, col_data)

            # if min(col_data) >= 0:
            #     col_values = col_data.values
            #     log_values = np.log1p(col_values)
            #     sqrt_values = np.sqrt(col_values)

            #     fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            #     sns.histplot(col_values, ax=ax[0], color='r')
            #     ax[0].set_xlim([min(col_values), max(col_values)])

            #     sns.histplot(log_values, ax=ax[1], color='r')
            #     ax[1].set_xlim([min(log_values), max(log_values)])

            #     sns.histplot(sqrt_values, ax=ax[2], color='r')
            #     ax[2].set_xlim([min(sqrt_values), max(sqrt_values)])
            #     plt.show()

            # if col_meta['dtype'][:3] == 'Num':
            #     dataset = convert_scale_domain(config, info, dataset, metaset)

            # # if info['dtype'][:3] == 'Cat':
            # #     dataset = convert_score_domain(info, dataset, metaset)

            # answer = request_user_input()

            # if answer:
            #     change_options = request_user_input(
            #         "how do you want to change this column ( [b:Bool, s:score] )",
            #         valid_inputs=['b', 's'], valid_outputs=['b', 's'], default='N'
            #     )
            #     if change_options == 'b':
            #         convert_boolean_domain(col, dataset)
            #     elif change_options == 's':
            #         convert_score_domain(col, dataset)

        return self.dataset


def print_meta_info(col_meta, col_data):
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
        print(col_data.describe(percentiles=[.03, .25, .50, .75, .97]))
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