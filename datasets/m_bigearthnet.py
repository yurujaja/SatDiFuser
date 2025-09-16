
import os
import random
import numpy as np
import torch
from torchvision import transforms
from typing import Dict, Tuple, Any

import geobench



class BigEarthNet:
    def __init__(self, cfg: Dict[str, Any], split: str, augment: bool = True):

        geobench_path = os.getenv("GEO_BENCH_DIR")
        if geobench_path is None:
            raise ValueError("Environment variable GEO_BENCH_DIR is not set.")
        
        self.dataset_path = os.path.join(geobench_path, 'classification_v1.0', 'm-bigearthnet')
        self.img_size = cfg['img_size']
        self.split = split
        self.augment = augment

        # Map split names to directory names, similar to your original class
        self.split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
        
        task = geobench.load_task_specs(self.dataset_path)
        self.dataset = task.get_dataset(split=self.split_mapping[self.split])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]
        rgb_bands = ("04", "03", "02")
        image, band_names = sample.pack_to_3d(band_names=rgb_bands)

        label = sample.label
        filename = sample.sample_name
        
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = image / 4095
        image = np.clip(image, 0, 1)
        image = torch.tensor(image)
        if self.augment and self.split == 'train':
            if random.random() > 0.5:
                augment_transoforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                ])
                image = augment_transoforms(image)
        
        image_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize([0.5], [0.5]),
        ])
        image = image_transform(image)

        output = {
            'rgb': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'filename': filename,
            'metadata': {}
        }
        return output

    @staticmethod
    def get_splits(cfg: Dict[str, Any]) -> Tuple['BigEarthNet', 'BigEarthNet', 'BigEarthNet']:

        dataset_train = BigEarthNet(cfg, split="train")
        dataset_val = BigEarthNet(cfg, split="val")
        dataset_test = BigEarthNet(cfg, split="test")
        return dataset_train, dataset_val, dataset_test
    
