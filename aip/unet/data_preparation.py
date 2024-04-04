#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:30:31 2019

@author: Jose Castillo 
"""
#%%    
import glob
import os
import numpy as np


def create_px_path(px_list):
    
    px_fol   = '/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed' 
    
    x_imag = []
    y_segm = []
    
    for p in px_list:
        images  = glob.glob(px_fol + '/'+p+'/images/*.nii')
        segmen  = glob.glob(px_fol+ '/'+p+'/segmentations/*.nii')
        x_imag  = x_imag+images 
        y_segm  = y_segm+segmen
        


    
    return x_imag, y_segm

#%% Generate Patients splits
from sklearn.model_selection import ShuffleSplit
    
def split_data_train_val_test(px_fol):

    patients = os.listdir(px_fol)
    patients = np.asarray(patients)

    ss    = ShuffleSplit(n_splits=1,test_size=0.20)
    ss.get_n_splits(patients)
    for train_index, test_index in ss.split(patients):
        xt, x_test = patients[train_index], patients[test_index]
        
    ss = ShuffleSplit(n_splits=1, test_size=0.20)
    ss.get_n_splits(xt)    
    for ten_index, val_index in ss.split(xt):
        x_train_in, x_val_in= xt[ten_index], xt[val_index]       
   
    px_splits = {'train': np.ndarray.tolist(x_train_in),
                 'val'  : np.ndarray.tolist(x_val_in)  ,
                 'test' : np.ndarray.tolist(x_test)    }
    
    return px_splits

#%% 