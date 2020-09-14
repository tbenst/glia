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
    nTimeSteps, nUnitChannels, meaH, meaW = retina_dset[0].shape
    image_dset = datamodule.images_dset
    imageChannels, H, W = image_dset[0].shape

    nCelltypes = trial.suggest_int("nCelltypes", 1, 32)
    conv1_out = trial.suggest_int("conv1_out", 1, 64)
    conv2_out = trial.suggest_int("conv2_out", 1, 64)
    kernel1 = trial.suggest_int("kernel1", 1, 16)*2-1 # must be odd
    kernel2 = trial.suggest_int("kernel2", 1, 16)*2-1 # must be odd
    kernel3 = trial.suggest_int("kernel3", 1, 16)*2-1 # must be odd
    nonlinearity = trial.suggest_categorical("nonlinearity",
        ["relu", "celu", "sigmoid", "tanh", "leaky_relu", "gelu", "hardswish"])
    batch_size =  trial.suggest_categorical("batch_size",
        [1,4,8,16,32,64,128,256,512])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4,1.)

    lr = trial.suggest_loguniform("lr", 1e-8,1.)
    beta = trial.suggest_loguniform("beta", 1e-3,1e3)
    
    hostname = socket.gethostname()

    model = ConvDecoder(H=H, W=W, imageChannels=imageChannels,
        nTimeSteps=nTimeSteps, 
        nCelltypes=nCelltypes, conv1_out=conv1_out, conv2_out=conv2_out,
        kernel1=kernel1, kernel2=kernel2, kernel3=kernel3,
        nUnitChannels=nUnitChannels, meaH=meaH, meaW=meaW,
        nonlinearity=nonlinearity, lr=lr, beta=beta,
        weight_decay=weight_decay,
        save_dir=save_dir, hostname=hostname, batch_size=batch_size)
    datamodule.batch_size = batch_size
    return model, datamodule

@dataclass(unsafe_hash=True)
class ConvDecoder(pl.LightningModule):
    "Given retina firing rates, reconstruct image"
        
    def __init__(self, hostname:str, save_dir:str, imageChannels:int = 3,
            nUnitChannels:int = 6, meaH:int = 64, meaW:int = 64, H:int = 64,
            W:int = 64, nTimeSteps:int = 10, nCelltypes:int = 8,
            conv1_out:int = 8, conv2_out:int = 8, kernel1:int = 15,
            kernel2:int = 15, kernel3:int = 15, lr:float = 1e-3,
            batch_size:int = 64, weight_decay: float = 1e-3, beta:float = 1,
            nonlinearity:Callable = F.sigmoid):
        super().__init__()
        self.hostname = hostname
        self.save_dir = save_dir
        self.imageChannels = imageChannels
        self.nUnitChannels = nUnitChannels
        self.meaH = meaH
        self.meaW = meaW
        self.H = H
        self.W = W
        self.nTimeSteps = nTimeSteps
        self.nCelltypes = nCelltypes
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.beta = beta
        self.nonlinearity = nonlinearity

        # all init args saved as hyperparam
        self.save_hyperparameters()

        if self.nonlinearity=="relu":
            self.nonlinearity = F.relu
        elif self.nonlinearity=="celu":
            self.nonlinearity = F.celu
        elif self.nonlinearity=="sigmoid":
            self.nonlinearity = F.sigmoid
        elif self.nonlinearity=="tanh":
            self.nonlinearity = F.tanh
        elif self.nonlinearity=="leaky_relu":
            self.nonlinearity = F.leaky_relu
        elif self.nonlinearity=="gelu":
            self.nonlinearity = F.gelu
        elif self.nonlinearity=="hardswish":
            self.nonlinearity = F.hardswish
        
        # TODO: loss of 1 - l2 norm on weights to encourage winner-takes-all
        cl = torch.FloatTensor(
            self.nCelltypes, self.nUnitChannels, self.meaH, self.meaW).fill_(
            1/self.nCelltypes)
        self.celltype_likelihood = nn.Parameter(torch.tensor(cl)) # uniform prior
        self.register_parameter(name='celltype', param=self.celltype_likelihood)

        self.conv1 = nn.Conv2d(in_channels=self.nCelltypes*self.nTimeSteps,
            out_channels=self.conv1_out, kernel_size=self.kernel1,
            padding=self.kernel1//2)
        self.conv2 = nn.Conv2d(in_channels=self.conv1_out,
            out_channels=self.conv2_out, kernel_size=self.kernel2,
            padding=self.kernel2//2)
        self.conv3 = nn.Conv2d(in_channels=self.conv2_out,
            out_channels=self.imageChannels, kernel_size=self.kernel3,
            padding=self.kernel3//2)
        
        for l in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(l.weight)
            nn.init.kaiming_normal_(self.celltype_likelihood)
    
    def forward(self,x):
        "Retina -> Image response"
        # map each unit to canonical celltype
#         prob = F.softmax(self.celltype_likelihood, dim=0)
        x = torch.einsum("btchw,nchw->btnhw", x, self.celltype_likelihood)
        # combine time and unit dimensions
        x = x.view(-1,self.nTimeSteps*self.nCelltypes, self.meaH, self.meaW)
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        return x
        
    def calc_loss(self, batch, batch_idx):
        bz = batch[0].shape[0]
        images, retina = batch
        im_pred = self(retina)
        recon_loss = F.mse_loss(im_pred, images, reduction='sum')/bz
        # encourage celltype prob to collapse onto single type
#         celltype_loss = (1 - F.softmax(self.celltype_likelihood,dim=0).norm(dim=0)).mean()
        celltype_loss = (1 - self.celltype_likelihood.norm(dim=0)).mean()
        loss = self.beta * celltype_loss + recon_loss
        return {"loss": loss, "recon_loss": recon_loss, "celltype_loss": celltype_loss,
                "im_pred": im_pred}

    def training_step(self, batch, batch_idx):
        loss_dict = self.calc_loss(batch, batch_idx)
        loss, recon_loss, celltype_loss = loss_dict["loss"], \
            loss_dict["recon_loss"], loss_dict["celltype_loss"]
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log('reconstruction_loss', recon_loss, prog_bar=True)
        result.log('celltype_loss', celltype_loss, prog_bar=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        loss_dict = self.calc_loss(batch, batch_idx)
        loss, recon_loss, celltype_loss, Y_pred = loss_dict["loss"], \
            loss_dict["recon_loss"], loss_dict["celltype_loss"], loss_dict["im_pred"]
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_epoch=True)
        result.log('val_mse_loss', recon_loss, on_epoch=True)
        result.log('val_celltype_loss', celltype_loss, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        return optimizer
    
    def on_sanity_check_end(self):
        Path(self.save_dir).mkdir(exist_ok=True)