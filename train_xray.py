#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:11:42 2020

@author: danbiderman
Train a DR-VAE, or just a VAE, on chest x-ray data.
When I want to test multiple betas, take a look at synthetic_experiment.py
"""
import torch
#print(torch.__version__)
import torchvision
#print(torchvision.__version__)
import torchxrayvision as xrv
import numpy as np
#import matplotlib.pyplot as plt
#import commentjson
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
parser.add_argument('--num_epochs', type=int, default=160, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--num_latents', type=int, default=100, help='')
parser.add_argument('--beta', type=float, default= 0.0, help='')
parser.add_argument('--dataset_size', type=int, default=None, help='')
parser.add_argument('--log_interval', type=int, default=None, help='')
parser.add_argument('--ignore_warnings', action="store_true")
parser.add_argument('--scale_down_image_loss', action="store_true")
parser.add_argument('--vae_only', action="store_true")
parser.add_argument("--recon_like_function", default='gaussian', help='image_loss')


args, _ = parser.parse_known_args()

# ToDo -- add a condition where it's just vanilla vae.? or probably not necessary if beta
# defaults to zero. add the torch xrv model, inspect the outputs and make sure
# it's fine.

class XRayResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        '''img: [None, img.shape[0], img.shape[1]] array'''
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            img = img[0,:,:]
            return cv2.resize(img, (self.size, self.size), 
                         interpolation = cv2.INTER_AREA).reshape(
                                1,self.size,self.size).astype(
                                    np.float32)

output_dir = os.path.join("./vae-xray", args.training_outcome)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define an image transformation for the dataset (in addition to torchxrayvision)
# note, didn't xrv's XRayResizer.
transform = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(),
      XRayResizer(args.image_size)]) # typically 224

# initialize a torchxrayvision dataset instance
d_kaggle = xrv.datasets.Kaggle_Dataset(
    imgpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_images_jpg'),
    csvpath=os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'stage_2_train_labels.csv'),
    dicomcsvpath = os.path.join(parent_dir, 'xray-datasets', 'kaggle-pneumonia-jpg',
                          'kaggle_stage_2_train_images_dicom_headers.csv.gz'),
    transform=transform)

# if train on test on a subset of the data
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
                                  input_dim=np.array([1, args.image_size, args.image_size]), # was np.array([1, 223, 223])
                                  n_ae_latents=args.num_latents,
                                  check_memory=False)
# add a couple of entries
arch_dict["model_class"] = 'vae'
arch_dict['ae_decoding_final_nonlin'] = 'clamp' # [str] 'linear' | 'sigmoid', if image is on [0,1] | 'clamp'
arch_dict['clamp_minmax'] = [-1.0,1.0]

if args.vae_only == True: # if just vae
    print('fitting just VAE.')
    model = ConvVAE(arch_dict, 
              scale_pixels = True,
              loglike_function = args.recon_like_function)
else:
    print('fitting a DR-VAE model.')
    # load discriminator, send to cuda, and set to eval mode (no dropout etc)
    discriminator = xrv.models.DenseNet(weights="all").cuda().eval()
    # freeze discriminator weights.
    for param in discriminator.parameters():
            param.requires_grad = False
    # define DRVAE model
    model = ConvDRVAE(arch_dict, 
                  scale_pixels = True,
                  loglike_function = args.recon_like_function)
    model.set_discrim_model(discriminator, 
                            discrim_beta = args.beta,
                            dim_out_to_use=8)

print('Built model.')
print(model.__str__())

print('check if cuda.is_available():')
print(torch.cuda.is_available())

# ToDo -- note which input you're feeding
resdict = {}
rundict = model.fit([None, None, None], 
        [None, None, None], 
        dataset = dataset, # note the dataset input.
        epochs = args.num_epochs,
        log_interval = args.log_interval,
        epoch_log_interval = 10,
        plot_interval = 10,
        output_dir = "vae-xray", # ToDo adapt
        torch_seed= int(0),
        batch_size=args.batch_size,
        scale_down_image_loss = args.scale_down_image_loss
        )

resdict['model'] = model


# # for inspiration, delete later.
# mlp_cycle_vae = LinearCycleVAE(n_channels=n_channels,
#                                        n_samples=n_samples,
#                                        hdims      = [500],
#                                        latent_dim = K)
# mlp_cycle_vae.set_discrim_model(mlp_model, discrim_beta=beta)
# mlp_cycle_vae.fit(Xdata, Xdata,
#                   epochs=num_epochs,
#                   log_interval=None,
#                   epoch_log_interval = 25,
#                   lr=lr,
#                   output_dir = vae_output_dir)
# resdict['beta-%s'%str(beta)] = mlp_cycle_vae

with open(os.path.join(output_dir, "resdict.pkl"), 'wb') as f:
            pickle.dump(resdict, f)
with open(os.path.join(output_dir, "rundict.pkl"), 'wb') as f:
            pickle.dump(rundict, f)

# another option - maybe easier to load on CPU
#torch.save(vae, join(cfg.output_dir, f'{dataset_name}-best.pt'))









