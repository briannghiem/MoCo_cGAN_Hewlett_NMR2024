import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, Reduction

class Config(object):
    def __init__(self,model_type,dataset='CC',depth=3,filters=[512,256,128,64]):
        # Define save directory
        origin = os.getcwd()

        self.save_dir_all = os.path.join(origin,'savedModels')
        self.save_dir = os.path.join(self.save_dir_all,model_type)
        
        # Miscellaneous
        self.summary = 0 # have tensorflow print out model summary
        self.strategy = tf.distribute.MirroredStrategy()
        self.loss_names = ['Discriminator loss','cGAN total loss',
                           'cGAN generator loss','cGAN discriminator loss']
        
        # Data info
        data_path = r'/home/nghiemb/Data/CC/simulated_datasets/MoCo_cGAN_Hewlett_NMR2024'
        self.data = data(data_path,dataset,model_type)        
        
        # Configure parameters
        self.training = training(model_type)        
        self.generator = generator(self.data.img_shape[3],depth,filters,self.data.complex_flag)        
        self.discriminator = discriminator(self.data.img_shape)        
        self.cGAN = cGAN(self.data.complex_flag)
      
        # Saving/loading the model
        self.summary_dir = origin
        self.load = loading()
        
class data(object):
    def __init__(self,origin,dataset,model_type):
        # Model variation
        typeSplit = model_type.split('_')
        if typeSplit[-1] == 'complex':
            self.complex_flag = True # complex input, represented in real network as two input channels
            typeSplit = typeSplit[:-1]
        else:
            self.complex_flag = False
        typeSplit = '_'.join(typeSplit)
        
        # Define data shape
        xdim = 224
        ydim = 192
        n_slices = 8        
        if typeSplit == 'cGAN':
            self.channel_opt = 'COMBINED'
            n_channels = 1
        if typeSplit == 'cGAN_singlechannel':
            self.channel_opt = 'SINGLE'
            n_channels = 1
        if typeSplit == 'cGAN_multichannel':
            self.channel_opt = 'ALL'
            n_channels = 28
        
        self.dir = origin
        self.dataset_name = dataset
        #
        self.ch_per_im = 1 # can select 1 to limit training to 1 channel per image, else use 'ALL'
        self.img_shape = [xdim,ydim,n_slices,n_channels*(self.complex_flag + 1)]
        self.img_shape_combined = [xdim,ydim,n_slices,self.complex_flag + 1]
        self.fraction = 1 # [0,1] can be used to limit datasets for debugging/testing
        self.contrast = 'T1' # use 'T2', 'T1', or 'FLAIR' to limit to specific contrast, else use 'ALL'

class training(object):
    # Configure optimization/training info
    def __init__(self,model_type):
        self.num_epochs = 100
        self.learning_rate = 5e-5 #default param stated in Hewlett et al
        
        # Set batch size
        if model_type == 'cGAN_multichannel' or model_type == 'cGAN_multichannel_complex':
            self.batch_size = 1
        else:
            self.batch_size = 4

        # Set number of checkpoints to keep during training
        self.ckpts_to_keep = None # Saves all checkpoints 

class generator(object):
    # Configure CNN architecture
    def __init__(self,n_channels,depth=3,filters=[512,256,128,64],complex_flag=False):
        self.out_ch = n_channels
        self.filters = filters
        self.depth = depth
        self.activation = 'relu'
        self.dropout = 0.5
        self.batchnorm = True
        self.maxpool = True
        self.upconv = True
        self.residual = True
        
        if complex_flag:
            self.out_activation = 'tanh'
        else:
            self.out_activation = 'sigmoid'         
        
class discriminator(object):
    # Configure CNN architecture
    def __init__(self,img_dim):
        self.loss_func = BinaryCrossentropy(reduction=Reduction.NONE)
        self.stride = 2
        self.kernel_size = 4
        self.bn_axis = -1
        
        # Define convolutional filters
        num_filters_i = 32
        nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
        self.filters_list = [num_filters_i * min(8, (2 ** i)) for i in range(nb_conv)]
        self.filters_list = self.filters_list[1:]     

class cGAN(object):
    # Configure cGAN
    def __init__(self,complex_flag=False):
        self.loss_func_d = BinaryCrossentropy(reduction=Reduction.NONE)
        self.loss_func_g = MeanAbsoluteError(reduction=Reduction.NONE)
        self.loss_weights = [4,1]
      
class loading(object):
    def __init__(self):
        self.opt = 1
        self.checkpoint = 'latest'