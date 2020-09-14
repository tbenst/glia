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

    n_hidden = trial.suggest_int("n_hidden", 5, 1000)
    n_hidden2 = trial.suggest_int("n_hidden2", 5, 1000)
    n_latent = trial.suggest_int("n_latent", 2, 100)
    nonlinearity = trial.suggest_categorical("nonlinearity",
        ["relu", "celu", "sigmoid", "tanh", "leaky_relu", "gelu", "hardswish"])
    loss_func =  trial.suggest_categorical("loss_func",
        ["mse", "bce"])
    batch_size =  trial.suggest_categorical("batch_size",
        [1,4,8,16,32,64,128,256,512])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4,1.)
    if loss_func=="mse":
        final_nonlinearity = trial.suggest_categorical("final_nonlinearity",
            [True, False])
    else:
        final_nonlinearity = F.sigmoid
    lr = trial.suggest_loguniform("lr", 1e-8,1.)
    beta = trial.suggest_loguniform("beta", 1e-3,1e3)
    
    hostname = socket.gethostname()

    model = VAE(n_input, H, W, n_hidden, n_latent, n_hidden2, n_output,
        nonlinearity=nonlinearity, lr=lr, beta=beta, loss_func=loss_func,
        final_nonlinearity=final_nonlinearity, weight_decay=weight_decay,
        save_dir=save_dir, hostname=hostname, batch_size=batch_size)
    datamodule.batch_size = batch_size
    return model, datamodule

class VAE(pl.LightningModule):
    
    def __init__(self, n_input, H, W, n_hidden, n_latent, n_hidden2, n_output,
                 weight_decay, save_dir, hostname, batch_size,
                 nonlinearity=F.sigmoid, beta=1, lr = 1e-4,
                 final_nonlinearity=True, loss_func='bce'):
        super().__init__()
        self.n_input = n_input
        self.H = H
        self.W = W
        self.beta = beta
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.n_latent = n_latent

        self.weight_decay = weight_decay
        if loss_func == "bce":
            self.loss_func = F.binary_cross_entropy
        else:
            self.loss_func = F.mse_loss
        if nonlinearity=="relu":
            f_nonlinear = F.relu
        elif nonlinearity=="celu":
            f_nonlinear = F.celu
        elif nonlinearity=="sigmoid":
            f_nonlinear = F.sigmoid
        elif nonlinearity=="tanh":
            f_nonlinear = F.tanh
        elif nonlinearity=="leaky_relu":
            f_nonlinear = F.leaky_relu
        elif nonlinearity=="gelu":
            f_nonlinear = F.gelu
        elif nonlinearity=="hardswish":
            f_nonlinear = F.hardswish
        if loss_func =="bce":
            self.final_nonlinearity = F.sigmoid
        elif final_nonlinearity:
            self.final_nonlinearity = f_nonlinear
        else:
            self.final_nonlinearity = lambda x: x

        self.nonlinearity = f_nonlinear
        self.lr = lr
                
        # Encoder layers
        self.hidden_encoder = nn.Linear(n_input, n_hidden)
        # mean encoding layer 
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        # log variance encoding layer 
        self.logvar_encoder = nn.Linear(n_hidden, n_latent)
        
        # Decoder layers
        self.hidden_decoder = nn.Linear(n_latent, n_hidden2)
        self.reconstruction_decoder = nn.Linear(n_hidden2, n_output)
        
        # init for logging images
        # make grid of z1 x z2 where z1,z2 \elem (-3.5,-2.5, ..., 3.5)
        # self.nrow = nrow

        self.save_dir = save_dir
        
        # all init args saved as hyperparam
        self.save_hyperparameters()
        
        self.recent_val_loss = 0.0
        
        # self.latents = torch.zeros(self.nrow,self.nrow,n_latent)#.cuda()
        # z1_tick = np.linspace(-3.5,3.5,self.nrow,dtype=np.float32)
        # z2_tick = np.linspace(-3.5,3.5,self.nrow, dtype=np.float32)
        # for i, z1 in enumerate(z1_tick):
        #     for j, z2 in enumerate(z2_tick):
        #         self.latents[i,j,[0,1]] = torch.tensor([z1,z2])
                
    def encode(self, x):
        h1 = self.nonlinearity(self.hidden_encoder(x))
        return self.mean_encoder(h1), self.logvar_encoder(h1)

    def reparameterize(self, mean, logvar):
        """Reparameterize out stochastic node so the gradient can propogate 
           deterministically."""

        if self.training:
            standard_deviation = torch.exp(0.5*logvar)
            # sample from unit gaussian with same shape as standard_deviation
            epsilon = torch.randn_like(standard_deviation)
            return epsilon * standard_deviation + mean
        else:
            return mean

    def decode(self, z):
        h3 = self.nonlinearity(self.hidden_decoder(z))
        return self.final_nonlinearity(self.reconstruction_decoder(h3))

    
    def forward(self, x):
        "A special method in PyTorch modules that is called by __call__"
        mean, logvar = self.encode(x)
        # sample an embedding, z
        z = self.reparameterize(mean, logvar)
        # return the (sampled) reconstruction, mean, and log variance
        return self.decode(z), mean, logvar

    
    def loss_function(self, recon_y, y, mu, logvar):
        "Reconstruction + KL divergence losses summed over all elements and batch."
        recon = self.loss_func(recon_y, y, size_average=False)

        # we want KLD = - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # where sigma is standard deviation and mu is mean
        # (see Appendix B of https://arxiv.org/abs/1312.6114)

        # see https://openreview.net/forum?id=Sy2fzU9gl for info on choosing Beta

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * KLD

    def calc_loss(self, batch, batch_idx):
        bz = batch[0].shape[0]
        # flatten batch x height x width x channel into batch x nFeatures
        images = batch[0].reshape(bz, -1)
        retina = batch[1].reshape(bz, -1) # flatten
        images_pred, mu, logvar = self(retina)
        loss = self.loss_function(images_pred, images, mu, logvar)
        return {"loss": loss, "Y": images, "Y_pred": images_pred,
                "X": retina}
        
    def training_step(self, batch, batch_idx):     
        loss_dict = self.calc_loss(batch, batch_idx)
        loss, Y, Y_pred = (loss_dict["loss"], loss_dict["Y"], loss_dict["Y_pred"])
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        bz = batch[0].shape[0]
        images = batch[0].reshape(bz, -1)
        retina = batch[1].reshape(bz, -1) # flatten
        images_pred, mu, logvar = self(retina)
        mse = F.mse_loss(images, images_pred, reduction="sum")/bz
        result = pl.EvalResult(checkpoint_on=mse)
        # MSE loss per image
        result.log('val_mse_loss', mse, on_step=False, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        return optimizer
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1120
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=self.lr, steps_per_epoch=len(data_loader), epochs=10)
        # scheduler = {"scheduler": scheduler, "interval" : "step" }
        # # steps_per_epoch = (train_loader_len//self.batch_size)//self.trainer.accumulate_grad_batches
        # return [optimizer], [scheduler]
    
    def on_sanity_check_end(self):
        Path(self.save_dir).mkdir(exist_ok=True)