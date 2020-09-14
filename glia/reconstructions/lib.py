
from typing import Tuple
import torch, tables
from torch import nn
import os
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from functools import partial
from tqdm.notebook import tqdm
from typing import Any
import torchvision
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
from dataclasses import dataclass
from typing import Callable
import cv2, functools, glia
from pytorch_lightning.loggers.neptune import NeptuneLogger
import neptune
import neptunecontrib.monitoring.optuna as optuna_utils
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import optuna
from datetime import datetime
import socket, plotly
from pathlib import Path
import torch.nn, gc, psycopg2
import sys, traceback

def resize_3d(images, fx, fy, interpolation=cv2.INTER_LINEAR, out="ndarray",
        is_tiff=False):
    im = cv2.resize((images[0]), None, fx=fx, fy=fy, interpolation=interpolation)
    if out=="ndarray":
        new = np.zeros([images.shape[0], *im.shape],
            dtype = np.float32)
    elif out=="memmap":
        new = np.memmap("glia_temp_memmap.mmap", np.float32, "w+",
            (images.shape[0], im.shape[0],im.shape[1]))
    for z, img in enumerate(images):
        new[z] = cv2.resize((img), None, fx=fx, fy=fy,
            interpolation=interpolation)
    return new

def gray_3d(images):
    im = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    new = np.zeros([images.shape[0], *im.shape],
            dtype = np.float32)
    for z, img in enumerate(images):
        new[z] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return new



@dataclass
class ImageData(Dataset):
    original_images:tables.carray.CArray
#     h5_path:str
#     dset_path:str = "images"
    H:int = 64 # resize to this
    W:int = 64
    convert_to_gray:bool = True
    zero_to_one:bool = True
    interpolation:Callable = cv2.INTER_LINEAR
    crop:Any = None
    
    def __post_init__(self):
        oH, oW = self.original_images.shape[1:3]
        crop = self.crop
        if crop:
            images = self.original_images[:,crop[0]:crop[1],crop[2]:crop[3]]
            oH, oW = images.shape[1:3]
        else:
            images = self.original_images
        self.images = resize_3d(images,
            fx=self.W/oW, fy=self.H/oH, interpolation=self.interpolation)
        if self.convert_to_gray:
            self.images = gray_3d(self.images)[...,None]
        if self.zero_to_one:
            self.images /= 255

        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        im = transforms.functional.to_tensor(self.images[i])
        return im
    
@dataclass
class RetinaData(Dataset):
    # we use HDF5 for compression of sparse data
    # otherwise, RAM would blow up. But after binning, not so large
    retina_data:tables.carray.CArray
    nbins:int = 10
        # collapse units into channel
    no_units:bool = False
    def __post_init__(self):
        slices = np.linspace(0, self.retina_data.shape[1], self.nbins+1, True).astype(np.int)
        bin_func = partial(np.add.reduceat, indices=slices[:-1], axis=0, dtype=np.float32)
        # THWC
        binned = np.array(glia.pmap(bin_func, self.retina_data, progress=True))
        # TCHW
        binned = torch.from_numpy(np.moveaxis(binned,-1,2))
        if self.no_units:
            binned = binned.sum(2, keepdims=True)
        self.data = binned
        
    def __len__(self):
        return self.retina_data.shape[0]
    def __getitem__(self,i):
        return self.data[i]


class ConcatDataset(torch.utils.data.Dataset):
    "Combine multiple datasets as tuples."
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


@dataclass
class ImageRetina(pl.LightningDataModule):
    images_dset:Dataset
    retina_dset:Dataset
    batch_size:int = 64
    # must be single-thread or else BLOSC barfs
    num_workers:int = 8
    tvt_idxs:Tuple=()

    def __post_init__(self):
        "tvt_idxs: len 3 tuple of indices for train, val, test split."
        super().__init__()
        concat_dataset = ConcatDataset(
            self.images_dset, self.retina_dset
        )
        N = len(self.images_dset)
        ntrain = int(N*0.8)
        nval = int(N*0.1)
        ntest = N-ntrain-nval
        lengths = [ntrain, nval, ntest]
        if len(self.tvt_idxs)>0:
            self.train = torch.utils.data.Subset(concat_dataset, self.tvt_idxs[0])
            self.val = torch.utils.data.Subset(concat_dataset, self.tvt_idxs[1])
            self.test = torch.utils.data.Subset(concat_dataset, self.tvt_idxs[2])
        else:
            self.train, self.val, self.test = random_split(concat_dataset, lengths,
                   generator=torch.Generator().manual_seed(42))
        
    def train_dataloader(self):
        
        loader = torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader