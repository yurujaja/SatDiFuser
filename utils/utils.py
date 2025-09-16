import torch
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf


def load_config(path):
    cfg = OmegaConf.load(path)
    
    if "base" in cfg:
        base_files = cfg["base"]
        base_cfgs = [OmegaConf.load(f"configs/{base}") for base in base_files]
        cfg = OmegaConf.merge(*base_cfgs, cfg) 
    
    return cfg


def combine_configs(args, cfg):
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    args_dict = {key: value for key, value in vars(args).items() if value is not None}
    
    args_conf = OmegaConf.create(args_dict)
    combined_config = OmegaConf.merge(cfg, args_conf)
    
    return combined_config


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
