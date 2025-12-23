import os
from torch.utils.tensorboard import SummaryWriter
import json


class Logger:
    def __init__(self, experiment_name, env_name):
        self.experiment_name = experiment_name
        self.env_name = env_name
        self.log_dir = os.path.join("runs", experiment_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        print(f"Logging to: {self.log_dir}")
    
    def log_hyperparameters(self, hparams):
        hparams_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=4)
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def close(self):
        self.writer.close()