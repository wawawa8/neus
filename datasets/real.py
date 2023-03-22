import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from models.ray_utils import get_ray_directions
from models.ray_utils import get_rays


class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = _get_rank()
        self.use_mask = True

        W, H = 4000, 6000
        self.img_downscale = 4.0
        self.near, self.far = 0.2, 4.0

        # W, H = 800, 800
        # self.img_downscale = 1
        # self.near, self.far = 2, 6

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        w, h = int(W // self.img_downscale), int(H // self.img_downscale)

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)




        self.all_c2w, self.all_images, self.all_fg_masks, self.all_rays_o, self.all_rays_d = [], [], [], [], []

        for i, key in enumerate(meta['frames']):
        # for i, frame in enumerate(meta['frames']):
            frame = meta['frames'][key]
            self.focal = 0.5 * w / math.tan(0.5 * frame['camera_angle_x']) # scaled focal length
            # self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

            # ray directions for all pixels, same for all images (same H, W, focal)
            self.directions = \
                get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2, self.config.use_pixel_centers).to(self.rank) # (h, w, 3)

            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            rays_o, rays_d = get_rays(self.directions, c2w.to(self.rank))
            self.all_rays_o.append(rays_o.view(self.h, self.w, -1))
            self.all_rays_d.append(rays_d.view(self.h, self.w, -1))

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_rays_o, self.all_rays_d = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_rays_o, dim=0).float().to(self.rank), \
            torch.stack(self.all_rays_d, dim=0).float().to(self.rank)


class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender-real')
class BlenderRealDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
