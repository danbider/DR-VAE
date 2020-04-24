#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:05:32 2020

@author: danbiderman
"""

import pickle
import os
import torch
import numpy as np

def load_object(filename):
    with open(filename, 'rb') as input: # note rb and not wb
        return pickle.load(input)
    
res_pickle = load_object(os.path.join('vae-xray', 'vanilla-vae', 'resdict.pkl'))

# access an example weight
list(res_pickle['vae'].encoding_net.encoder.modules())[2].weight

# push forward a random batch of data
res_pickle['vae'].forward(torch.tensor(np.random.normal(size=(32,1,64,64))*1024.0, 
                                       dtype = torch.float))
