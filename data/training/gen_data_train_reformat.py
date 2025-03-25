'''
From Hewlett et al, NMR Biomed 2024
Main file for training a cGAN for motion correction in MRI.

Modified March 18, 2025
Retraining on T1w MPRAGE data from Calgary-Campinas dataset, with simulated motion corruption
'''

import os
import numpy as np

#-------------------------------------------------------------------------------
# # LOADING DATA

#load and prep data
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN'
dpath = mpath + r'/data/training_dataset/slices'
# spath = mpath + r'/networks'

spath = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024' #

#Loading complex-valued data
train_labels = np.load(dpath + r'/train/train_GT.npy') #dim: [3072,8,192,224,2]
train_corr = np.load(dpath + r'/train/train_corr.npy') #corr for corrupted

val_labels = np.load(dpath + r'/val/val_GT.npy') #dim: [768,8,192,224,2]
val_corr = np.load(dpath + r'/val/val_corr.npy') #corr for corrupted

#Transpose data to align with Johnson & Drangova's convention
train_labels = np.transpose(train_labels, axes=(3,2,1,4,0)) #dim: [224,192,8,2,3072]
train_corr = np.transpose(train_corr, axes=(3,2,1,4,0)) #dim: [224,192,8,2,3072]

val_labels = np.transpose(val_labels, axes=(3,2,1,4,0)) #dim: [224,192,8,2,768]
val_corr = np.transpose(val_corr, axes=(3,2,1,4,0)) #dim: [224,192,8,2,768]

n_train = train_labels.shape[-1]
n_val = val_labels.shape[-1]

#-------------------------------------------------------------------------------
#Splitting the data into magnitude and phase, as needed by Hewlett et al
#Generate dictionaries, and save samples individually
keys = ['ImageMotionArray', 'ImageArray', 'ImageMotionArrayPHASE', 'ImageArrayPHASE']

#-------------------------------
#For training dataset
for i in range(n_train):
    print('Saving Training Sample {}'.format(i+1), end='\r')
    spath_temp = spath + r'/training/train_sample{}.npy'.format(i+1)
    train_label_temp = train_labels[...,0,i] + 1j*train_labels[...,1,i]
    train_corr_temp = train_corr[...,0,i] + 1j*train_corr[...,1,i]
    #
    dict_temp = {keys[0]:np.abs(train_corr_temp[..., None]),\
                    keys[1]:np.angle(train_label_temp[..., None]),\
                    keys[2]:np.abs(train_corr_temp[..., None]),\
                    keys[3]:np.angle(train_label_temp[..., None])}
    #
    np.save(spath_temp, dict_temp)

#-------------------------------
#For validation dataset
for i in range(n_val):
    print('Saving Training Sample {}'.format(i+1), end='\r')
    spath_temp = spath + r'/validation/val_sample{}.npy'.format(i+1)
    val_label_temp = val_labels[...,0,i] + 1j*val_labels[...,1,i]
    val_corr_temp = val_corr[...,0,i] + 1j*val_corr[...,1,i]
    #
    dict_temp = {keys[0]:np.abs(val_corr_temp[..., None]),\
                    keys[1]:np.angle(val_label_temp[..., None]),\
                    keys[2]:np.abs(val_corr_temp[..., None]),\
                    keys[3]:np.angle(val_label_temp[..., None])}
    #
    np.save(spath_temp, dict_temp)

