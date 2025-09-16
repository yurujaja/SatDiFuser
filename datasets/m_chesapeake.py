
import os
import random
import numpy as np
import torch
from torchvision import transforms
from typing import Dict, Tuple, Any
from PIL import Image
import geobench



class ChesaPeake:
    def __init__(self, cfg: Dict[str, Any], split: str, augment: bool = True):
        geobench_path = os.getenv("GEO_BENCH_DIR")
        if geobench_path is None:
            raise ValueError("Environment variable GEO_BENCH_DIR is not set.")
        
        self.dataset_path = os.path.join(geobench_path, 'segmentation_v1.0', 'm-chesapeake')
        self.img_size = cfg['img_size']
        self.split = split
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.1)


        # Map split names to directory names, similar to your original class
        self.split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
        
        task = geobench.load_task_specs(self.dataset_path)
        self.dataset = task.get_dataset(split=self.split_mapping[self.split])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]
        rgb_bands = ("Red", "Green", "Blue")
        image, band_names = sample.pack_to_3d(band_names=rgb_bands)

        label = sample.label.data
        filename = sample.sample_name
        
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = np.clip(image, 0, 1)
        image = torch.tensor(image)
        
        # image_8bit = (image * 255).astype(np.uint8)
        # image_8bit = np.transpose(image_8bit, (1, 2, 0))
        # img_pil = Image.fromarray(image_8bit, mode='RGB')
        # label_pil = Image.fromarray(label.astype(np.uint8), mode='L')
                
        # if self.augment and self.split == 'train':
        #     if random.random() > 0.5:
        #         img_pil = F.hflip(img_pil)
        #         label_pil = F.hflip(label_pil)
        #     if random.random() > 0.5:
        #         img_pil = F.vflip(img_pil)
        #         label_pil = F.vflip(label_pil)
        #     if random.random() > 0.5:    
        #         img_pil = self.color_jitter(img_pil)
        
        image_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize([0.5], [0.5]),
        ])
        image = image_transform(image)
        label = torch.tensor(label, dtype=torch.int64)

        output = {
            'rgb': image,
            'label': label,
            'filename': filename,
            'metadata': {}
        }
        return output

    @staticmethod
    def get_splits(cfg: Dict[str, Any]) -> Tuple['ChesaPeake', 'ChesaPeake', 'ChesaPeake']:
        dataset_train = ChesaPeake(cfg, split="train")
        dataset_val = ChesaPeake(cfg, split="val")
        dataset_test = ChesaPeake(cfg, split="test")
        return dataset_train, dataset_val, dataset_test
    
