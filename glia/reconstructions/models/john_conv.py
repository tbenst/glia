import torch, tables
from torch import nn
import os
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from functools import partial
from tqdm.auto import tqdm
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

# Kingma & Welling (2014) style variational autoencoder

# subclass PyTorch Module for reverse-mode autodifferentiation 
# for easy backpropogation of loss gradient

def sample_model(trial, datamodule, save_dir):
    retina_dset = datamodule.retina_dset
    image_dset = datamodule.images_dset
    n_input = np.prod(retina_dset[0].shape)
    n_output = np.prod(image_dset[0].shape)
    H, W = image_dset[0].shape[1:]

    filters = trial.suggest_int("filters", 4,256)
    nLayers = trial.suggest_int("nLayers", 2,151)
    kernel1 = trial.suggest_int("kernel1", 1, 8)*2-1
    nonlinearity = trial.suggest_categorical("nonlinearity",
        ["relu", "celu", "sigmoid", "tanh", "leaky_relu", "gelu", "hardswish"])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5,1e-1)
    batch_size=64
    poisson = trial.suggest_categorical("poisson",
        [True, False])
    einsum = trial.suggest_categorical("einsum",
        [True, False])
    if einsum:
        nCelltypes = trial.suggest_int("nCelltypes", 4,32)
    else:
        nCelltypes = 0
    model = ConvNet(filters=filters, nLayers=nLayers, kernel1=kernel1, nCelltypes=nCelltypes,
        weight_decay=weight_decay, poisson=poisson, einsum=einsum,
        example_input_array=retina_dset[0][None])
    datamodule.batch_size = batch_size
    return model, datamodule

class Block(torch.nn.Module):
    def __init__(self, filters):
        super(Block, self).__init__()
        
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters), torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters))
        
    def forward(self, x):
        return F.relu(x + self.block(x))
    
class ConvNet(pl.LightningModule):
    def __init__(self, filters=128, nLayers=101, lr=3e-4, kernel1=9, weight_decay=0.,
                poisson=False, einsum=False, nCelltypes=0,
                example_input_array=None):
        super(ConvNet, self).__init__()
        self.filters = filters
        self.lr = lr
        self.weight_decay=weight_decay
        
        self.save_hyperparameters()
        if einsum:
            self.nCelltypes = nCelltypes
        else:
            self.nCelltypes = 6
        self.conv_in = torch.nn.Conv2d(10*self.nCelltypes, filters, kernel1, padding=4,
            stride=4, bias=False)
        
        self.resnet = torch.nn.ModuleList()
        for i in range(nLayers): self.resnet.append(Block(filters))
            
        self.conv_out =  torch.nn.ConvTranspose2d(self.filters, 1, 8, stride=8, bias=True)
        self.bias = torch.nn.Parameter(torch.Tensor(64,64))
        
        for name, parm in self.named_parameters():
            if name.endswith('weight'): torch.nn.init.normal_(parm, 0, .01)
            if name.endswith('bias'): torch.nn.init.constant_(parm, 0.0)

        cl = torch.FloatTensor(
        self.nCelltypes, 6, 64, 64).fill_(
            1/self.nCelltypes)
        self.celltype_likelihood = nn.Parameter(torch.tensor(cl)) # uniform prior
        self.register_parameter(name='celltype', param=self.celltype_likelihood)
    
    def forward(self, x):
        if self.poisson:
            x = torch.poisson(x)
        if self.einsum:
            x = torch.einsum("btchw,nchw->btnhw", x, self.celltype_likelihood)
        x = x.reshape(-1,10*self.nCelltypes,64,64)[:,:,::2,::2].contiguous() # factor of 2 downsampling
        zx = F.relu(self.conv_in(x))
        for layer in self.resnet: zx = layer(zx)
        return torch.sigmoid(self.conv_out(zx).squeeze() + self.bias[None,:,:])
    
    def loss(self, batch, batch_idx):        
        images, retina = batch
        retina = retina/30 # scale 
        images = images[:,0] # remove singleton
        im_pred = self(retina)
#         loss = F.mse_loss(im_pred, images, reduction='mean')
        loss = ((im_pred - images)**2).sum(1).mean()
        recon_loss = F.mse_loss(im_pred, images, reduction='sum') / images.shape[0]
        return {"recon_loss": recon_loss, "loss": loss}
        
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch, batch_idx)
        loss, recon_loss = loss_dict["loss"], loss_dict["recon_loss"]
        result = pl.TrainResult(loss)
        result.log('reconstruction_loss', recon_loss, prog_bar=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        loss_dict = self.loss(batch, batch_idx)
        loss, recon_loss = loss_dict["loss"], loss_dict["recon_loss"]
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_mse_loss', recon_loss, on_epoch=True)
        return result

    def test_step(self, batch, batch_idx):
        loss_dict = self.calc_loss(batch, batch_idx)
        loss, recon_loss = loss_dict["loss"], loss_dict["recon_loss"]
        return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        return optimizer
