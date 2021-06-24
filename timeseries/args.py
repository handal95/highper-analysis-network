import yaml

class Args:
    def __init__(self, file, train=False):
        option = "train" if train else "test"

        with open(file) as f:
            config = yaml.safe_load(f)["args"][option] 
           
        self.workers = config["workers"]
        self.batch_size = config["batch_size"]
        
        if train:
            self.epochs = config["epochs"]
            self.lr = config["lr"]
            self.cuda = config["cuda"]
            self.manualSeed = config["manualSeed"]
        