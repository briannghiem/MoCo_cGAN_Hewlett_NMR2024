'''
From Hewlett et al, NMR Biomed 2024
Main file for training a cGAN for motion correction in MRI.

Created March 25, 2025
Reformatting data created for SAPUNet / UNetJE evaluation for Paper 1
*Need to reformat dataset to be able to load into in cGAN NN
'''

import os
import numpy as np
import pathlib as plib

#-------------------------------------------------------------------------------
# # HELPER FUNCTIONS

def load_m_GT(dpath_temp_root):
    m_GT_init = np.load(dpath_temp_root + r'/current_test_GT.npy') #SI, LR, AP
    m_GT = np.pad(m_GT_init[:,:,:,0,0] + 1j*m_GT_init[:,:,:,1,0], ((1,1), (0,0), (0,0)))
    del m_GT_init
    m_GT = np.transpose(m_GT, (1,2,0))[6:-6,3:-3,:]
    return m_GT

def load_m_temp(dpath_temp_root):
    m_temp = np.load(dpath_temp_root + r'/m_corrupted.npy')
    n_LR = m_temp.shape[0]
    if n_LR == 170:
        m_temp = np.pad(m_temp, ((5,5),(0,0),(0,0)))
    return m_temp

def image_transform(image, target_shape, slab_size):
    #Reformat volumes into slabs
    image_transpose = np.transpose(image, axes=(1,0,2)) #new shape: 218, 180, 256
    image_pad = np.pad(image_transpose, ((3,3), (6,6), (0,0))) #new shape: 224, 192, 256
    #
    reshape_size = (target_shape[2]//slab_size,slab_size,target_shape[0],target_shape[1])
    image_transpose2 = np.transpose(image_pad, axes=(2,0,1))
    image_reshape = image_transpose2[None,...].reshape(reshape_size) #new shape: 32, 8, 224, 192
    image_out = np.transpose(image_reshape, axes=(2,3,1,0))
    return image_out

def make_dict(m_corrupted, m_GT, keys):
    dict_temp = {keys[0]:np.abs(m_corrupted[..., None]),\
                    keys[1]:np.abs(m_GT[..., None]),\
                    keys[2]:np.angle(m_corrupted[..., None]),\
                    keys[3]:np.angle(m_GT[..., None])}
    return dict_temp


#-------------------------------------------------------------------------------
# # LOADING DATA

#load and prep data
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN_Hewlett_NMR2024'
spath = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024/data/testing' #

target_shape = [224,192,256]
slab_size = 8
keys = ['ImageMotionArray', 'ImageArray', 'ImageMotionArrayPHASE', 'ImageArrayPHASE']

#Loading data from PyMoCo test data (from Paper 1)
dpath_init = r'/home/nghiemb/PyMoCo/data/cc/test/PE1_AP/Complex/R1'
paradigms = [r'/Paradigm_1C', r'/Paradigm_1D', r'/Paradigm_1E', r'/Paradigm_1F']
subs = [1,4,5,6,7]

for paradigm in paradigms:
    print("DATASET: {}".format(paradigm))
    #
    for sub in subs:
        print("Subject {}".format(sub))
        dpath_temp_root = dpath_init + paradigm + r'/Test{}'.format(sub)
        spath_temp_root = spath + r'/{}'.format(paradigm) + r'/Test{}'.format(sub)
        plib.Path(spath_temp_root).mkdir(parents=True, exist_ok=True)
        #Loading data
        m_GT = load_m_GT(dpath_temp_root) #shape: 180, 218, 256
        m_temp = load_m_temp(dpath_temp_root) #shape: 180, 218, 256
        #Reformatting into slabs of 8 axial slices
        m_GT = image_transform(m_GT, target_shape, slab_size) #shape: 224, 192, 8, 32
        m_temp = image_transform(m_temp, target_shape, slab_size) #shape: 224, 192, 8, 32
        #
        for slab in range(m_temp.shape[-1]):
            print("Slab: {}".format(slab+1), end = '\r')
            spath_temp = spath_temp_root + r'/slab{}.npy'.format(slab+1)
            dict_temp = make_dict(m_temp[..., slab], m_GT[..., slab], keys)
            np.save(spath_temp, dict_temp)





'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_views(img, vmax = 1.0):
    if vmax == "auto": #if auto, set as max val of volume
        vmax = abs(img.flatten().detach().cpu()).max()
    #
    fig, axes = plt.subplots(1,3)
    for i, ax in enumerate(axes):
        if i==0:
            ax.imshow(img[img.shape[0]//2,:,:], cmap = "gray", vmax = vmax)
        if i==1:
            ax.imshow(img[:,img.shape[1]//2,:], cmap = "gray", vmax = vmax)
        if i==2:
            ax.imshow(img[:,:,img.shape[2]//2], cmap = "gray", vmax = vmax)
        #
    plt.show()

'''
