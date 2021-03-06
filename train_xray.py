#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:11:42 2020

@author: danbiderman
Train a DR-VAE, or just a VAE, on chest x-ray data.
When I want to test multiple betas, take a look at synthetic_experiment.py
"""
# Install torchxrayvision package from Dan's github, see https://stackoverflow.com/questions/15268953/how-to-install-python-package-from-github
import os
os.system('pip install --upgrade git+https://github.com/danbider/torchxrayvision.git')
import torch
from torch import nn
import torchvision
import torchxrayvision as xrv # package that was just installed.
import numpy as np
from drvae.model.ae_model_architecture_generator import *
from drvae.model.vae import ConvVAE, ConvDRVAE
import argparse, os, shutil, time, sys, pyprind, pickle
import warnings
import inspect
import cv2
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--training-outcome", default='gaussian', help='what to predict')
parser.add_argument("--training", action="store_true")
parser.add_argument("--evaluating", action="store_true")
parser.add_argument('--image_size', type=int, default=224, help='')
parser.add_argument('--num_epochs', type=int, default=300, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_latents', type=int, default=100, help='')
parser.add_argument('--beta', type=float, default= 0.0, help='')
parser.add_argument('--dataset_size', type=int, default=None, help='')
parser.add_argument('--log_interval', type=int, default=None, help='')
parser.add_argument('--ignore_warnings', action="store_true")
parser.add_argument('--scale_down_image_loss', action="store_false")
parser.add_argument('--vae_only', action="store_true")
parser.add_argument("--recon_like_function", default='gaussian', help='image_loss')
parser.add_argument('--num_zero_kl_epochs', type=int, default=0, help='')
parser.add_argument('--anneal_rate', type=float, default= 0.02, help='')
parser.add_argument('--epoch_log_interval', type=int, default= 10, help='')

args, _ = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XRayResizer(object):
    '''A class that resizes images using cv2 instead of the torchxrayvision defaults'''
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        '''img: [None, img.shape[0], img.shape[1]] array'''
        img = img[0,:,:]
        resized = cv2.resize(img, (self.size, self.size), 
                     interpolation = cv2.INTER_AREA).reshape(
                            1,self.size,self.size).astype(
                                np.float32)
        if len(np.unique(resized)) < 2:
            sys.exit('dataloader warning: all pixel values are the same.')
        return resized

# output directory is a function of beta and num_latents.
# currently fixed to 'experiments-kaggle'. change for new datasets.
output_dir = os.path.join("./experiments-kaggle", 
                          "beta_%s_num_latents_%i" % (str(args.beta), args.num_latents))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define an image transformation for the dataset (in addition to torchxrayvision)
# note, we don't use xrv's XRayResizer.
transform = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), # # can change back to xrv.datasets if import package
      XRayResizer(args.image_size)]) # typically 224

# initialize a torchxrayvision dataset instance
# torchxrayvision map style datasset. Change the dataset object for different datasets.
d_kaggle = xrv.datasets.Kaggle_Dataset(
    imgpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_images_jpg'),
    csvpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_labels.csv'),
    dicomcsvpath = os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'kaggle_stage_2_train_images_dicom_headers.csv.gz'),
    transform=transform)

# if train and test on a smaller subset of the data, pick args.dataset_size elements randomly.
if args.dataset_size is not None:
    random_indices = np.random.choice(len(d_kaggle),
                                      size=args.dataset_size,
                                      replace=False)
    dataset = torch.utils.data.Subset(d_kaggle, random_indices)
    print('training-testing on a subset of %i images' % args.dataset_size)
else:
    dataset = d_kaggle

# make arch dict for VAE from ae_arch_new.json
arch_dict = load_handcrafted_arch(ae_arch_json=os.path.join(
    'json_configs', 'ae_arch_new.json'),
                                  input_dim=np.array([1, args.image_size, args.image_size]),
                                  n_ae_latents=args.num_latents,
                                  check_memory=False)

# define DRVAE model - if args.beta=0, it's just VAE but with an attached discriminator
model = ConvDRVAE(arch_dict,
                  scale_pixels = True, # this is specific to torchxrayvision dataset where images are in [-1024, 1024]
                  loglike_function = args.recon_like_function) # function for the likelihood of the datpoint.
discriminator = xrv.models.DenseNet(weights="all").to(device).eval() # can change to xrv.models in future
for param in discriminator.parameters():
         param.requires_grad = False # discriminator is fixed.
model.set_discrim_model(discriminator, 
                        discrim_beta = args.beta, 
                        dim_out_to_use=8, # 8 is for pneumonia in torchxrayvision
                        disc_output_type = 'probs') # output could also be logit
if args.beta == 0: # if just vae, previously args.vae_only=True
    print('fitting just VAE.')
else:
    print('fitting a DR-VAE model.')

print('Built model.')
print(model.__str__())

print('check if cuda.is_available():')
print(torch.cuda.is_available())

resdict = {}
rundict = model.fit([None, None, None], 
        [None, None, None], 
        dataset = dataset, # note the dataset input.
        epochs = args.num_epochs,
        log_interval = args.log_interval,
        epoch_log_interval = args.epoch_log_interval,
        plot_interval = 10,
        output_dir = output_dir,
        torch_seed= int(0),
        batch_size=args.batch_size,
        scale_down_image_loss = args.scale_down_image_loss,
        anneal_rate = args.anneal_rate,
        num_zero_kl_epochs = args.num_zero_kl_epochs,
        )

resdict['model'] = model
resdict['discrim_beta'] = args.beta
resdict['num_latents'] = args.num_latents

with open(os.path.join(output_dir, "resdict.pkl"), 'wb') as f:
            pickle.dump(resdict, f)
with open(os.path.join(output_dir, "rundict.pkl"), 'wb') as f:
            pickle.dump(rundict, f)

# another option - maybe easier to load on CPU
#torch.save(vae, join(cfg.output_dir, f'{dataset_name}-best.pt'))









