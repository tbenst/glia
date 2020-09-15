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

    # nCelltypes = trial.suggest_int("nCelltypes", 1, 32)
    nCelltypes=1
    kernel1 = trial.suggest_int("kernel1", 1, 8)*2-1 # must be odd
    kernel2 = trial.suggest_int("kernel2", 1, 8)*2-1 # must be odd
    nLayers = trial.suggest_int("nLayers", 1, 8)
    nBlockChannels = trial.suggest_int("nBlockChannels", 1, 64)
    nonlinearity = trial.suggest_categorical("nonlinearity",
        ["relu", "celu", "sigmoid", "tanh", "leaky_relu", "gelu", "hardswish"])
    batch_size =  trial.suggest_categorical("batch_size",
        [8,16,32,64,128,256,512])
    # batch_size = 64
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6,1.)

    # lr = trial.suggest_loguniform("lr", 1e-8,1.)
    # beta = trial.suggest_loguniform("beta", 1e-6,1e3)
    beta = 0
    
    hostname = socket.gethostname()

    model = ResnetDecoder(H=H, W=W, imageChannels=imageChannels,
        nTimeSteps=nTimeSteps, 
        nCelltypes=nCelltypes,
        kernel1=kernel1, kernel2=kernel2, nLayers=nLayers,
        nBlockChannels=nBlockChannels,
        nUnitChannels=nUnitChannels, meaH=meaH, meaW=meaW,
        nonlinearity=nonlinearity,beta=beta, weight_decay=weight_decay, no_eigsum=True,
        example_input_array=retina_dset[0][None], # single batch
        # lr=lr, 
        save_dir=save_dir, hostname=hostname, batch_size=batch_size)
    datamodule.batch_size = batch_size
    return model, datamodule

def convLayer(in_chan, out_chan, kernel=9):
    return nn.Conv2d(in_chan, out_chan, kernel_size=kernel,
                     padding=kernel//2, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, nonlinearity=F.relu):
        super(BasicBlock, self).__init__()
        self.conv1 = convLayer(in_chan, out_chan)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.nonlinearity = nonlinearity
        self.conv2 = convLayer(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.nonlinearity(out)

        return out


@dataclass(unsafe_hash=True)
class ResnetDecoder(pl.LightningModule):
    "Given retina firing rates, reconstruct image"
        
    def __init__(self, hostname:str, save_dir:str, example_input_array,
            imageChannels:int = 3,
            nUnitChannels:int = 6, nBlockChannels:int = 16,
            meaH:int = 64, meaW:int = 64, H:int = 64, nLayers:int = 4, no_eigsum:bool=False,
            W:int = 64, nTimeSteps:int = 10, nCelltypes:int = 8,
            kernel1:int = 15, kernel2:int = 15, lr:float = 1e-3,
            batch_size:int = 64, weight_decay: float = 1e-3, beta:float = 1,
            nonlinearity:Callable = F.sigmoid):
        super().__init__()
        self.hostname = hostname
        self.save_dir = save_dir
        self.imageChannels = imageChannels
        self.nUnitChannels = nUnitChannels
        self.nBlockChannels = nBlockChannels
        self.meaH = meaH
        self.meaW = meaW
        self.H = H
        self.W = W
        self.nTimeSteps = nTimeSteps
        self.nCelltypes = nCelltypes
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.nLayers = nLayers
        self.no_eigsum = no_eigsum
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.beta = beta
        self.nonlinearity = nonlinearity
        self.example_input_array = example_input_array

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
        self.celltype_likelihood = nn.Parameter(cl) # uniform prior
        if no_eigsum:
            conv1_input = self.nTimeSteps * self.nUnitChannels
        else:
            conv1_input = self.nTimeSteps * self.nCelltypes
        self.conv1 = nn.Conv2d(in_channels=conv1_input,
            out_channels=self.nBlockChannels, kernel_size=self.kernel1,
            padding=self.kernel1//2)
        self.bn1 = nn.BatchNorm2d(self.nBlockChannels)

        layers = [BasicBlock(self.nBlockChannels, self.nBlockChannels, self.nonlinearity)
                  for i in range(self.nLayers)]
        conv2 = nn.Conv2d(in_channels=self.nBlockChannels,
            out_channels=self.imageChannels, kernel_size=self.kernel2,
            padding=self.kernel2//2)
        layers.append(conv2)
        self.layers = nn.Sequential(*layers)
  
        self.save_hyperparameters()
    
    def forward(self,x):
        "Retina -> Image response"
        # map each unit to canonical celltype
#         prob = F.softmax(self.celltype_likelihood, dim=0)
        if self.training:
            x = torch.poisson(x)
        if not self.no_eigsum:
            x = torch.einsum("btchw,nchw->btnhw", x, self.celltype_likelihood)
            # combine time and unit dimensions
            x = x.view(-1,self.nTimeSteps*self.nCelltypes, self.meaH, self.meaW)
        else:
            x = x.reshape(-1, self.nUnitChannels*self.nTimeSteps, self.meaH, self.meaW)

        x = self.conv1(x)
        x = self.nonlinearity(self.bn1(x))
        x = self.layers(x)
        x = self.nonlinearity(x)
        return x
        
    def calc_loss(self, batch, batch_idx):
        images, retina = batch
        im_pred = self(retina)
        recon_loss = F.mse_loss(im_pred, images, reduction='sum') / images.shape[0]
        celltype_loss = (1 - self.celltype_likelihood.norm(dim=0)).mean()
        # TODO: WARN: beta is not used!
#         loss = self.beta * celltype_loss + recon_loss
        loss = recon_loss
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

    def test_step(self, batch, batch_idx):
        loss_dict = self.calc_loss(batch, batch_idx)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        return optimizer
