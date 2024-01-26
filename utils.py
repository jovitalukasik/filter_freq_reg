import torch
import argparse
from datetime import datetime
import os
import pandas as pd
import wandb
import numpy as np
import random
import logging

class CSVLogger:
    
    def __init__(self, log_file):
        self.rows = []
        self.log_file = log_file 
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, epoch, row, silent=False):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch,**row}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.log_file, index=False)


class ConsoleLogger:

    def log(self, epoch, row, silent=False):
        if not silent:
            logging.info(f"[{datetime.now()}] Epoch {epoch} - {row}")


class WandBLogger:
    
    def __init__(self, model, project, args, output_dir):
        wandb.init(config=args, project=project)
        wandb.run.name = output_dir
        wandb.watch(model, log="all")
        
    def log(self, epoch, row, silent=False):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch, **row}
        wandb.log(row)

    def log_plot(self, plot):
        wandb.log({"plot": plot})

        
class CheckpointCallback:  
    CKPT_PATTERN = "epoch=%d.ckpt"
    
    def __init__(self, path, mode="all", args=None):
        
        assert mode in ["all", None, 10]
        
        self.path = path 
        self.mode = mode
        self.args = args
        
        os.makedirs(self.path, exist_ok=True)

    def save(self, epoch,  model, metrics, force=False, name=None):
        if self.mode == "all" or force:
            if name:
                out_path = os.path.join(self.path, "epoch="+name+".ckpt")
            else:
                out_path = os.path.join(self.path, self.CKPT_PATTERN % (epoch))
            logging.debug(f"saving {out_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(), 
                    "metrics": {"epoch": epoch,**metrics}, 
                    "args": self.args
                }, out_path)
        if self.mode == 10:
            if (epoch+1) % 10 == 0 or epoch == 0:
                out_path = os.path.join(self.path, self.CKPT_PATTERN % (epoch))
                logging.debug(f"saving {out_path}")
                torch.save(
                    {
                        "state_dict": model.state_dict(), 
                        "metrics": {"epoch": epoch,**metrics}, 
                        "args": self.args
                    }, out_path)
    
    
def none2str(value): 
    if value == "None":
        return None
    return value


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of "--arg1 true --arg2 false"
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def prepend_key_prefix(d, prefix):
    return dict((prefix + key, value) for (key, value) in d.items())


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed = seed


def get_arg(args, key, fallback=None):
    if key in vars(args):
        return vars(args)[key]
    return fallback

def step(e): 
    e = e.split("/")[-1]
    if e[e.rfind("epoch=")+6:e.rfind(".")] == "best":
        return int(-1)
    if e[e.rfind("epoch=")+6:e.rfind(".")]== "last_adv":
        return int(-2)
    return int(e[e.rfind("epoch=")+6:e.rfind(".")])

