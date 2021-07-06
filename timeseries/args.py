import argparse
import yaml

class Args:
    def __init__(self):
        parser = self.get_parser()
        self.args = parser.parse_args()
        
        print(self.args)
        
            
    def get_parser(self):
        parser = argparse.ArgumentParser(
            add_help=False, description="Command Line Interface")
        parser.set_defaults(function=None)
        parser.add_argument(
            "--data", type=str, default="config/data_config.yml", help="data.yml path")
        
        return parser
    
    def get_option(self, train):
        option = "train" if train else "test"

        with open(self.args.data) as f:
            config = yaml.safe_load(f)["args"][option] 
           
        self.workers = config["workers"]
        self.batch_size = config["batch_size"]
        
        if train:
            self.epochs = config["epochs"]
            self.lr = config["lr"]
            self.cuda = config["cuda"]
            self.manualSeed = config["manualSeed"]        
    
        