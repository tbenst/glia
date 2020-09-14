# import hdf5plugin, h5py
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
import pytorch_lightning as pl
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
from glia.reconstructions.lib import resize_3d, gray_3d, ImageData, RetinaData, \
     ConcatDataset, ImageRetina
from sqlalchemy import create_engine

filepath = sys.argv[1]
# model_base_dir = "/storage/models/retina-reconstruction"
model_base_dir = sys.argv[2]
gpus = [ int(sys.argv[3]) ]

pw = os.environ["POSTGRES_OPTUNA_PASSWORD"]
server = os.environ["POSTGRES_SERVER"]
port = os.environ["POSTGRES_PORT"]
user = os.environ["POSTGRES_OPTUNA_USER"]
# validate password

engine = create_engine(f'postgresql://{user}:{pw}@{server}:{port}/optuna')
with engine.connect() as connection:
    result = connection.execute("select * from trials limit 1;")


# open file
hdf5 = tables.open_file(filepath,'r')
h5 = hdf5.root

# crop image
imgs = h5["images"][0:100]
avg_img = np.mean(imgs, axis=(0,3))
y,x = np.where(avg_img>5)
xleft = min(x)
xright = max(x)
ytop = min(y)
ybot = max(y)

image_dset = ImageData(h5["images"], crop=[ytop,ybot,xleft,xright])
retina_dset = RetinaData(h5["data"], no_units=False)
(image_dset[0].shape, retina_dset[0].shape)

# TODO add args
MODEL_NAME = "VAE"
from models.vae import sample_model


def objective(trial, tags, save_dir, max_train_iter, datamodule,
        monitor='val_mse_loss', gpus=[0]):
    "Optuna objective"
    # gc.collect()
    # torch.cuda.empty_cache()
    save_dir = os.path.join(save_dir, f"trial_{trial.number}")
    model = sample_model(trial, datamodule, save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=save_dir,
        save_top_k=True,
        verbose=False,
        monitor=monitor,
        mode='min',
        prefix=''
    )

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="tbenst/retina",
        params=model.hparams,
        experiment_name="VAE",  # Optional,
        tags=["optuna-trial"] + tags
    )
    trainer = pl.Trainer(gpus=gpus, gradient_clip_val=0.5,
        logger=neptune_logger, checkpoint_callback=checkpoint_callback,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor=monitor),
        max_epochs=max_train_iter)
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#weights-save-path
    #     weights_save_path=save_dir)
    #                      logger=mlf_logger)
    # trainer = pl.Trainer(num_processes=1, gradient_clip_val=0.5)
    trainer.fit(model, dm)
    
    return trainer.logged_metrics[monitor]

nSamples = len(retina_dset)
train_idx = [i for i in range(nSamples) if (i+1)%5 != 0]
val_idx = np.arange(nSamples)[slice(4,nSamples,10)]
test_idx = np.arange(nSamples)[slice(9,nSamples,10)]
dm = ImageRetina(image_dset, retina_dset,batch_size=64,
                 tvt_idxs=(train_idx, val_idx, test_idx))

assert os.path.exists(model_base_dir)
now_str = datetime.now().isoformat()
save_dir = os.path.join(model_base_dir,
    now_str + "-optuna")

neptune.init(project_qualified_name='tbenst/retina')

tags = ["VAE", now_str+"-optuna"]
neptune.create_experiment('optuna', tags=["optuna-master"] + tags)
neptune_callback = optuna_utils.NeptuneCallback()

max_train_iter = 25
study = optuna.create_study(direction='minimize',
   pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=max_train_iter,
        reduction_factor=3
    ),
    storage=f'postgresql://{user}:{pw}@{server}:{port}/optuna'
)
    
# catch RuntimeError ie CUDA error: device-side assert triggered
# TODO not a good idea...?
study.optimize(partial(objective, tags=tags, save_dir=save_dir,
        max_train_iter=max_train_iter, datamodule=dm, gpus=gpus),
    callbacks=[neptune_callback],
    timeout=60*60*8, gc_after_trial=True)
optuna_utils.log_study(study)