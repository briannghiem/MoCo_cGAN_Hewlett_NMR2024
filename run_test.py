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
# os.environ['CUDA_VISIBLE_DEVICES'] = '' #turn off GPU for debugging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))

#-------------------------------------------------------------------------------
# HELPER FUNCTIONS

def slab2volume(slabs):
    return np.squeeze(tf.concat(slabs, 2).numpy())


#-------------------------------------------------------------------------------
# SETTING UP MODEL

# model_type = 'cGAN_complex'
model_type = 'cGAN'

config = Config(model_type)

#Loading pre-trained model
config.data.dir = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024'

config.load.opt = 1 # load saved models
# config.load.checkpoint = '99'
# config.load.checkpoint = '39'
config.load.checkpoint = '13'

config.training.num_epochs = int(config.load.checkpoint) - 1 # skip training and stick with saved model

model = cGAN(config)

#-------------------------------------------------------------------------------
# # LOADING TEST DATA
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN_Hewlett_NMR2024'
dpath = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024/data/testing' #

paradigms = [r'/Paradigm_1C', r'/Paradigm_1D', r'/Paradigm_1E', r'/Paradigm_1F']
subs = [1,4,5,6,7]

paradigm = paradigms[0] #choose motion level
sub = subs[0] #choose test subject
dpath_identifiers = r'/testing' + paradigm + r'/Test{}'.format(sub)
dpath_temp_root = dpath + paradigm + r'/Test{}'.format(sub)

# Preparing test data
test_data = prepare_test_data(config, dpath_temp_root, datatype=dpath_temp_root)

# Evaluate cGAN on test data
from time import time
t1 = time()
groundtruth_store, corrupted_store, corrected_store = model.eval_test(test_data)
t2 = time()
print("Elapsed time for 1 volume: {:.2f} sec".format(t2 - t1)) #for 32 slabs: 6.2 seconds on GPU, 33.7 seconds on CPU

groundtruth_array = slab2volume(groundtruth_store)
corrupted_array = slab2volume(corrupted_store)
corrected_array = slab2volume(corrected_store)

plot_views(abs(groundtruth_array))
plot_views(abs(corrupted_array))
plot_views(abs(corrected_array))


'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_views(img, vmax = 1.0):
    if vmax == "auto": #if auto, set as max val of volume
        # vmax = abs(img.flatten().detach().cpu()).max()
        vmax = abs(img.flatten()).max()
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


ind_slab = 16
temp_GT = groundtruth_store[ind_slab].numpy()
temp_corrupted = corrupted_store[ind_slab].numpy()
temp_output = corrected_store[ind_slab].numpy()

plot_views(np.abs(temp_GT))
plot_views(np.abs(temp_corrupted))

plot_views(np.abs(temp_output))

'''
