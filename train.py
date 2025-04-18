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
# model_type = 'cGAN' 

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

config.data.dir = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024'



training_continue = 1 #flag for continue training from checkpoint
if training_continue:
    config.load.checkpoint = '39'
else:
    config.load.opt = 0 # load saved models (turn off to train from scratch)


# Configure model
model = cGAN(config)

#-------------------------------------------------------------------------------
# # LOADING DATA

# Set up datasets
train_data = prepare_train_data(config,'training')
validation_data = prepare_train_data(config,'validation')

# # TRAINING
model.train(train_data=train_data, val_data=validation_data)

# # Sample output
# model.sample_output(validation_data,'Validation Example',config.training.num_epochs)

'''
#Check loss curves

import numpy as np
import math
import matplotlib.pyplot as plt

#-------------------------------
#Helper functions
def get_ymax(train_array, val_array):
    ymax_init = np.max((train_array[0], val_array[0]))
    ymax = 10**(math.ceil(math.log(ymax_init, 10)))
    return ymax

def get_ymin(train_array, val_array):
    ymin_init = np.min((train_array[-1], val_array[-1]))
    ymin = 10**(math.floor(math.log(ymin_init, 10)))
    return ymin

def get_curves(train_array, val_array, y_axis_title, ylims = 'default'):
    epoch_array = np.arange(1,len(train_array)+1)
    if ylims == 'default':
        ymin = np.min((min(train_array), min(val_array)))
        ymax = np.max((max(train_array), max(val_array)))
    elif ylims == 'cropped':
        ymin = get_ymin(train_array, val_array)
        ymax = get_ymax(train_array, val_array)
    else:
        ymin = ylims[0]
        ymax = ylims[1]
    #
    plt.figure()
    plt.plot(epoch_array, train_array, label = "training")
    plt.plot(epoch_array, val_array, label = "validation")
    plt.xlabel("Epoch")
    plt.ylabel(y_axis_title)
    plt.title("cGAN - Loss Curves; {}".format(y_axis_title))
    plt.ylim(ymin, ymax)
    plt.show()

#-------------------------------
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN_Hewlett_NMR2024'
spath = mpath + r'/savedModels/cGAN_complex/checkpoints'
# spath = mpath + r'/savedModels/cGAN/checkpoints'

loss = np.load(spath + r'/loss.npy', allow_pickle=1).item() #recover dictionary from 0-array

keys = list(loss.keys())

#-------------------------------
#Mean Absolute Error
MAE_train = loss[keys[-2]]['training']
MAE_val = loss[keys[-2]]['validation']

get_curves(MAE_train, MAE_val, 'MAE', ylims = 'default')

#Binary Cross Entropy
cGAN_CE_train = loss[keys[-1]]['training'] #cross entropy
cGAN_CE_val = loss[keys[-1]]['validation'] #cross entropy

get_curves(cGAN_CE_train, cGAN_CE_val, 'Cross Entropy', ylims = 'default')
get_curves(np.log(cGAN_CE_train)/np.log(10), np.log(cGAN_CE_val)/np.log(10), 'Cross Entropy - Log_10 plot', ylims = 'default')
# get_curves(np.log(cGAN_CE_train), np.log(cGAN_CE_val), 'Cross Entropy - Log_e plot', ylims = 'default')


#-------------------------------
#Get best iteration

MAE_train_min_ind = int(np.where(np.array(MAE_train) == np.array(MAE_train).min())[0])
MAE_val_min_ind = int(np.where(np.array(MAE_val) == np.array(MAE_val).min())[0])
print("MAE_train minimum of {:.3f} at iteration {}".format(np.array(MAE_train).min(), MAE_train_min_ind))
print("MAE_val minimum of {:.3f} at iteration {}".format(np.array(MAE_val).min(), MAE_val_min_ind))

cGAN_CE_train_min_ind = int(np.where(np.array(cGAN_CE_train) == np.array(cGAN_CE_train).min())[0])
cGAN_CE_val_min_ind = int(np.where(np.array(cGAN_CE_val) == np.array(cGAN_CE_val).min())[0])
print("cAGN_CE_train minimum of {} at iteration {}".format(np.array(cGAN_CE_train).min(), cGAN_CE_train_min_ind))
print("cGAN_CE_val minimum of {} at iteration {}".format(np.array(cGAN_CE_val).min(), cGAN_CE_val_min_ind))

'''