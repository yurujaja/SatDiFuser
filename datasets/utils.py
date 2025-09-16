import importlib
from torch.utils.data import Dataset
import torch

DATASET_REGISTRY = {
    "meurosat": "m_eurosat.EuroSAT",
    "mbigearthnet": "m_bigearthnet.BigEarthNet",
    "mnz-cattle": "m_nz_cattle.NZCattle",
    "mchesapeake": "m_chesapeake.ChesaPeake",
    "mcashew": "m_cashew_plant.CashewPlant",
}


def get_dataset_class(dataset_name):
    """
    Retrieve dataset class from registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        valid_datasets = sorted(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not implemented. Valid datasets: {valid_datasets}"
        )
    
    module_path, class_name = DATASET_REGISTRY[dataset_name].rsplit(".", 1)
    module = importlib.import_module(f"datasets.{module_path}")
    return getattr(module, class_name)


def get_datasets(cfg):
    """
    Get train, validation, and test dataset splits based on config.
    """
    if not hasattr(cfg, "dataset_name") or not cfg.dataset_name:
        raise AttributeError("cfg must have a valid 'dataset_name' attribute")
    dataset_cls = get_dataset_class(cfg.dataset_name)
    try:
        splits = dataset_cls.get_splits(cfg)
        if not isinstance(splits, (tuple, list)) or len(splits) != 3:
            raise ValueError(
                f"{dataset_cls.__name__}.get_splits must return a tuple of (train, val, test)"
            )
        dataset_train, dataset_val, dataset_test = splits         
        return dataset_train, dataset_val, dataset_test
    
    except Exception as e:
        raise ValueError(f"Failed to get splits for dataset '{cfg.dataset_name}': {str(e)}")


class GeoBenchSubset(Dataset):
    def __init__(self, full_dataset, indices, repeat=1):
        self.full_dataset = full_dataset
        self.indices = indices
        self.repeat = repeat

    def __getitem__(self, idx):
        idx = self.indices[idx // self.repeat]
        return self.full_dataset[idx]

    def __len__(self):
        return len(self.indices) * self.repeat