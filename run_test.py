'''
From Hewlett et al, NMR Biomed 2024
Main file for training a cGAN for motion correction in MRI.

Modified March 18, 2025
Retraining on T1w MPRAGE data from Calgary-Campinas dataset, with simulated motion corruption
'''

#-------------------------------------------------------------------------------
from utils.cgan_ops import cGAN
from utils.dataset_prep import prepare_test_data
from utils.model_config import Config

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

#limit GPU memory usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


#-------------------------------------------------------------------------------
# SETTING UP MODEL

model_type = 'cGAN_complex'

config = Config(model_type)

#Loading pre-trained model
config.data.dir = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024'

config.load.opt = 1 # load saved models
config.load.checkpoint = '100'
config.training.num_epochs = int(config.load.checkpoint) - 1 # skip training and stick with saved model

model = cGAN(config)

#-------------------------------------------------------------------------------
# # LOADING TEST DATA
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN_Hewlett_NMR2024'
dpath = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024/data/testing' #

paradigms = [r'/Paradigm_1C', r'/Paradigm_1D', r'/Paradigm_1E', r'/Paradigm_1F']
subs = [1,4,5,6,7]

paradigm = paradigms[0]
sub = subs[0]
dpath_temp_root = dpath + paradigm + r'/Test{}'.format(sub)


# Set up datasets
test_data = prepare_test_data(config, dpath_temp_root)


# # Sample output
# model.sample_output(validation_data,'Validation Example',config.training.num_epochs)


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
