"""
Define the dataset class for the cGAN network

Based on code from https://github.com/nikhilmaram/Show_and_Tell/blob/master/dataset.py
"""

import numpy as np
import os
import random
import tensorflow as tf
import time
from utils.image_process import combine_coils, extract_complex, gen_complex, image_normalize

#-------------------------------------------------------------------------------
#Loading npy files with proper alphanumeric sorting
import os
import glob
import re

def atoi(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

#-------------------------------------------------------------------------------
def prepare_train_data(config,datatype):
    with config.strategy.scope():
        # Load the training/validation data    
        fname_temp = os.listdir(os.path.join(config.data.dir,datatype)) 
        image_files = sorted(fname_temp, key = natural_keys) #sort alphanumerical order
        # image_files = os.listdir(os.path.join(config.data.dir,datatype)) #get all image paths
        #
        # # Remove additional contrasts if not wanted
        # if not config.data.contrast == 'ALL':
        #     filelist_editor(image_files,'include','prefix','file_brain_AX' + config.data.contrast)
        #      
        # Remove additional samples if requested
        if not config.data.fraction == 1:
            req_size = round(config.data.fraction*len(image_files))
            random.seed(38)
            image_files = random.sample(image_files,req_size)
            random.seed()
        #  
        dataset = DataSet(image_files,
                            datatype,
                            config)
        #
        dataset.datatype = datatype
    #
    print("Dataset built")
    time.sleep(1)
    #
    return dataset

def prepare_test_data(config,test_folder,datatype='testing'):
    with config.strategy.scope():
        # Load test dataset from provided folder 
        fname_temp = os.listdir(test_folder) 
        image_files = sorted(fname_temp, key = natural_keys) #sort alphanumerical order  
        dataset = DataSet(image_files,
                            datatype,
                            config)
        #
        dataset.datatype = datatype
    #
    print("Dataset built")
    time.sleep(1)
    #
    return dataset

def filelist_editor(filelist,option,location,text):
    """
    This function will remove files from a list based on a prefix or suffix.
    #
    Inputs
        filelist: list of file names
        option: 'include' or 'exclude' files based on a condition
        location: 'prefix' or 'suffix'
        text: text of the prefix/suffix
    #
    Outputs:
        filelist: updated list of file names
    """
    #
    files_to_remove = []
    #
    for file in filelist:
        # Determine whether the file contains the prefix/suffix of interest
        if location == 'prefix':
            condition = file.startswith(text)
        if location == 'suffix':
            condition = file.endswith(text)
        #
        # Exclude files that do not meet the condition
        if option == 'exclude':
            if condition:
                files_to_remove.append(file)
        if option == 'include':
            if not condition:
                files_to_remove.append(file)
    #
    for file in files_to_remove:
        filelist.remove(file)

class DataSet(object):
    def __init__(self,
                 image_files,
                 datatype,
                 config):
        #
        # Get info about dataset
        self.image_count = len(image_files)
        self.complex_flag = config.data.complex_flag
        self.image_ids = np.array(range(self.image_count))
        self.image_files = image_files
        # self.n_channels = [int(image[-6:-4]) for image in image_files] #specific to their data file naming convention
        self.n_channels = [1 for image in image_files]
        #
        # Assign characteristics from config
        self.batch_size = config.training.batch_size
        self.dir = os.path.join(config.data.dir,datatype)
        self.img_shape = config.data.img_shape
        #
        # Prepare training samples
        self.channel_opt = config.data.channel_opt
        if self.channel_opt == 'COMBINED' or self.channel_opt == 'ALL':
            self.count = len(self.image_ids)
            self.file_identifier = self.image_files
            self.channel_identifier = [self.channel_opt for n in range(self.count)]
        elif self.channel_opt == 'SINGLE':
            if config.data.ch_per_im == 'ALL':
                self.count = sum(self.n_channels)
            else:
                self.count = config.data.ch_per_im*len(self.image_ids)
            #
            self.file_identifier = [None for n in range(self.count)]
            self.channel_identifier = [None for n in range(self.count)]
            idx_start = 0
            for ix, image in enumerate(self.image_files):
                if config.data.ch_per_im == 'ALL':
                    idx_end = idx_start + self.n_channels[ix]
                    channels = range(self.n_channels[ix])
                else:
                    idx_end = idx_start + config.data.ch_per_im
                    channels = random.sample(range(self.n_channels[ix]),config.data.ch_per_im)
                #
                for channel, idx in enumerate(range(idx_start,idx_end)):
                    self.file_identifier[idx] = self.image_files[ix]
                    self.channel_identifier[idx] = channels[channel] + 1
                #
                idx_start = idx_end
        #
        self.idxs = list(range(self.count))
    #
    def reset(self):
        np.random.shuffle(self.idxs)
    #
    def load(self,file,channel):
        path = os.path.join(self.dir,file)
        #
        data = np.load(path,allow_pickle=True).item()
        images_motion = tf.cast(data.get('ImageMotionArray'),dtype=tf.float32) #retrieve magnitude of sample
        images_corrected = tf.cast(data.get('ImageArray'),dtype=tf.float32) #retrieve magnitude of label
        #
        # # Normalize (performed on magnitude component)
        # images_motion,_,_ = image_normalize(images_motion) #### skipping, since all volumes were already normalized
        # images_corrected,_,_ = image_normalize(images_corrected) #### skipping, since all volumes were already normalized
        # #
        # Check for phase data
        if 'ImageArrayPHASE' in data:
            phase_motion = tf.cast(data.get('ImageMotionArrayPHASE'),dtype=tf.float32)
            phase_corrected = tf.cast(data.get('ImageArrayPHASE'),dtype=tf.float32)
            #
            # Convert to complex tensors
            images_motion = gen_complex(images_motion,phase_motion)
            images_corrected = gen_complex(images_corrected,phase_corrected)
            #
            if channel == 'COMBINED' or channel == 'SINGLE' or channel == 'ALL':
                # Check for CSM data
                if 'CSMsReal' in data:
                    CSMsReal = tf.cast(data.get('CSMsReal'),dtype=tf.float32)
                    CSMsImag = tf.cast(data.get('CSMsImag'),dtype=tf.float32)
                    CSMconj = tf.complex(CSMsReal,tf.math.scalar_mul(-1,CSMsImag))
                else:
                    CSMconj = []
            #
            else:
                CSMconj = []
        else:
            CSMconj = []
        #
        # Get desired channel
        if channel == 'COMBINED':
            # Combine channels
            # images_corrected, img_min, img_max = combine_coils(images_corrected,[],CSMsConj=CSMconj)
            # images_motion,_,_ = combine_coils(images_motion,[img_min, img_max],CSMsConj=CSMconj)
            # TEMPORARY PATCH - SKIPPING COIL COMBINATION
            images_motion = images_motion
            images_corrected = images_corrected   
        elif channel == 'SINGLE':
            # Return images as is
            images_motion = images_motion
            images_corrected = images_corrected     
        elif channel == 'ALL':
            # Determine how much padding is needed and what value should be used
            channels2pad = self.img_shape[3]/(self.complex_flag + 1) - images_corrected.shape[3]
            paddings = tf.cast(tf.constant([[0,0],[0,0],[0,0],[0,channels2pad]]),dtype=tf.int32)
            #
            if images_motion.dtype == tf.complex64:
                value = tf.complex(tf.constant(0,dtype=tf.float32),tf.constant(0,dtype=tf.float32))
            else:
                value = tf.constant(0,dtype=tf.float32)    
            #
            # Zero pad to expected number of channels
            images_motion = tf.pad(images_motion,paddings,"CONSTANT",constant_values=value)
            images_corrected = tf.pad(images_corrected,paddings,"CONSTANT",constant_values=value)
            #
            # Also pad CSMs for uniformity within DL networks
            if tf.is_tensor(CSMconj):
                CSMconj = tf.pad(CSMconj,paddings,"CONSTANT",constant_values=value)
        else:
            # Return specific channel
            images_motion = images_motion[:,:,:,channel - 1]
            images_corrected = images_corrected[:,:,:,channel - 1]
            #
            images_motion = tf.expand_dims(images_motion,axis=3)
            images_corrected = tf.expand_dims(images_corrected,axis=3)
        #
        # Delete CSMs if no longer necessary
        if not channel == 'ALL':
            CSMconj = []
        #
        if self.complex_flag:
            # Seperate into real and imaginary components
            images_motion = extract_complex(images_motion)
            images_corrected = extract_complex(images_corrected)
        else:
            # Take magnitude
            images_motion = tf.math.abs(images_motion)
            images_corrected = tf.math.abs(images_corrected)
        #
        return images_motion, images_corrected, CSMconj