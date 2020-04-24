#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:11:42 2020

@author: danbiderman
Train a DR-VAE, or just a VAE, on chest x-ray data.
When I want to test multiple betas, take a look at synthetic_experiment.py
"""
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
import commentjson
from drvae.model.ae_model_architecture_generator import *
from drvae.model.vae import ConvVAE
import argparse, os, shutil, time, sys, pyprind, pickle

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--training-outcome", default='gaussian', help='what to predict')
parser.add_argument("--training", action="store_true")
parser.add_argument("--evaluating", action="store_true")
parser.add_argument('--image_size', type=int, default=224, help='')
parser.add_argument('--num_epochs', type=int, default=160, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--num_latents', type=int, default=100, help='')
parser.add_argument('--dataset_size', type=int, default=None, help='')

args, _ = parser.parse_known_args()

output_dir = os.path.join("./vae-xray", args.training_outcome)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define an image transformation for the dataset.
transform = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(),
      xrv.datasets.XRayResizer(args.image_size)]) # final val be 224. shrink just for test ma

# define a torchxrayvision dataset
d_kaggle = xrv.datasets.Kaggle_Dataset(
    imgpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_images_jpg'),
    csvpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_labels.csv'),
    transform=transform)

# toy dataset
if args.dataset_size is not None:
    random_indices = np.random.choice(len(d_kaggle),
                                      size=args.dataset_size,
                                      replace=False)
    subset_dataset = torch.utils.data.Subset(d_kaggle, random_indices)
    print('training-testing on a subset of %i images' % args.dataset_size)

# make arch dict for VAE from ae_arch_new.json
arch_dict = load_handcrafted_arch(ae_arch_json=os.path.join(
    'json_configs', 'ae_arch_new.json'),
                                  input_dim=np.array([1, args.image_size, args.image_size]), # was np.array([1, 223, 223])
                                  n_ae_latents=args.num_latents,
                                  check_memory=False)
# add a couple of entries
arch_dict["model_class"] = 'vae'
arch_dict['ae_decoding_final_nonlin'] = 'sigmoid' # [str] 'linear' | 'sigmoid', if image is on [0,1]

vae = ConvVAE(arch_dict, 
              scale_pixels = True)

print('Built model.')
print(vae.__str__())

resdict = {}
vae.fit([None, None, None], 
        [None, None, None], 
        dataset = subset_dataset, # note the dataset input.
        epochs = args.num_epochs,
        log_interval = 2,
        epoch_log_interval = 4,
        plot_interval = 3,
        output_dir = "vae-xray",
        torch_seed= int(0),
        batch_size=args.batch_size,
        )
resdict['vae'] = vae

with open(os.path.join(output_dir, "resdict.pkl"), 'wb') as f:
            pickle.dump(resdict, f)

# another option.
#torch.save(vae, join(cfg.output_dir, f'{dataset_name}-best.pt'))









