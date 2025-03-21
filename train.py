'''
From Hewlett et al, NMR Biomed 2024
Main file for training a cGAN for motion correction in MRI.

Modified March 18, 2025
Retraining on T1w MPRAGE data from Calgary-Campinas dataset, with simulated motion corruption
'''

#-------------------------------------------------------------------------------
from utils.cgan_ops import cGAN
from utils.dataset_prep import prepare_train_data
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
'''
'cGAN_complex' (performs motion correction on complex coil-combined data)
'cGAN_singlechannel_complex' (performes motion correction on each complex channel independantly)
'cGAN_multichannel_complex' (performs motion correction on all complex channels simultaneously)
Each variation can also be trained to perform correction on magnitude data by removing '_complex'
'''

# Configure settings
if model_type[0:17] == 'cGAN_multichannel':
    config = Config(model_type,filters=[2048,1024,512,256])
else:
    config = Config(model_type)


config.load.opt = 0 # load saved models (turn off to train from scratch)
config.data.dir = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024'

# Configure model
model = cGAN(config)
# model.summary() #creates and saves model txt file in current working directory

'''
if config.load.opt:
    if model_type ==  'cGAN_complex':
        config.load.checkpoint = '73'
    elif model_type == 'cGAN_singlechannel_complex':
        config.load.checkpoint = '64'
    elif model_type == 'cGAN_multichannel_complex':
        config.load.checkpoint = '47'  
    config.training.num_epochs = int(config.load.checkpoint) - 1 # skip training and stick with saved model
'''

#-------------------------------------------------------------------------------
# # LOADING DATA

# Set up datasets
train_data = prepare_train_data(config,'training')
validation_data = prepare_train_data(config,'validation')

# # TRAINING
model.train(train_data=train_data)

# Sample output
model.sample_output(validation_data,'Validation Example',config.training.num_epochs)



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