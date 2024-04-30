import numpy as np
import tensorflow as tf
from utils.image_process import combine_coils, extract_complex, gen_complexRI
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Dense, Flatten, InputLayer
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.normalization import BatchNormalization

class CC_layer(tf.keras.layers.Layer):
    # Custom layer which performs coil combination
    def __init__(self,coil_dim,CSM_flag):
        super(CC_layer, self).__init__()
        self.coil_dim = coil_dim
        self.CSM_flag = CSM_flag
       
    def call(self,inputs):
        # Get inputs and convert to complex tensors for compatibility with coil combination function
        if self.CSM_flag:
            multichannelImg, CSMs = inputs
        else:
            multichannelImg = inputs
            CSMs = []
            
        if self.CSM_flag:
            # Change to complex tensor as expected by CC
            multichannelImg = gen_complexRI(multichannelImg,coil_dim=self.coil_dim)
        
        combined_image,_,_ = combine_coils(multichannelImg,-1,self.coil_dim,CSMsConj=CSMs)
        
        if self.CSM_flag:
            # Reformat in [Real,Imag] implementation
            combined_image = extract_complex(combined_image,coil_dim=self.coil_dim)
            
        return combined_image
       
class get_discriminator(object):
    def __init__(self,config):
        config.discriminator.img_shape = config.data.img_shape
        config.discriminator.img_shape_combined = config.data.img_shape_combined
        config.discriminator.channel_opt = config.data.channel_opt
        config.discriminator.complex_flag = config.data.complex_flag
        
        config.discriminator.csm_shape = config.discriminator.img_shape[0:3]
        if config.discriminator.complex_flag:  
            if np.mod(config.discriminator.img_shape[3],2):
                config.discriminator.csm_shape.append(-1) # Error
            else:
                config.discriminator.csm_shape.append(int(config.discriminator.img_shape[3]/2))          
        
        self.config = config.discriminator
   
    def build(self):
        tf.print("Building discriminator...")
        config = self.config
        
        # 'Model' coil combination
        if config.channel_opt == 'ALL':
            if config.complex_flag:
                # CSM-based coil combination
                ccInput = [Input(shape=config.img_shape,dtype=tf.float32,name="MultichannelImage"),Input(shape=self.config.csm_shape,dtype=tf.complex64,name="CSMs")]
                ccOutput = CC_layer(coil_dim=4,CSM_flag=True)(ccInput)
            elif not config.complex_flag:
                # RSS coil combination
                ccInput = Input(shape=config.img_shape,dtype=tf.float32,name="MultichannelImage")
                ccOutput = CC_layer(coil_dim=4,CSM_flag=False)(ccInput)        
        
        gan_discriminator = Sequential(name = 'discriminator')
        gan_discriminator.add(InputLayer(input_shape=config.img_shape_combined))        
        
        # Add the first convolutional layer; 4x4x4 filters with stride
        gan_discriminator.add(Conv3D(filters=64, kernel_size=config.kernel_size, padding='same',
                                 strides=(config.stride,config.stride,config.stride), name='disc_conv_1'))
        gan_discriminator.add(LeakyReLU(alpha=0.2))        
        
        # Add remaining convolutional layers; 4x4x4 filters with stride=2 and batch normalization
        for i, filter_size in enumerate(config.filters_list):
            name = 'disc_conv_{}'.format(i+2)
            gan_discriminator.add(Conv3D(filters=filter_size, kernel_size=config.kernel_size, padding='same',
                                         strides=(config.stride,config.stride,config.stride), name=name))
            gan_discriminator.add(BatchNormalization(name=name + '_bn', axis=config.bn_axis))
            gan_discriminator.add(LeakyReLU(alpha=0.2))        
        
        # Flatten and predict outcome with fully connected layer
        gan_discriminator.add(Flatten())
        gan_discriminator.add(Dense(2, activation='softmax', name="disc_dense"))
        
        # Combine with coil combination model for complex multichannel network
        if config.channel_opt == 'ALL':
            discOutput = gan_discriminator(ccOutput)
            discriminatorModel = Model(inputs=ccInput,outputs=discOutput,name = 'discriminatorCC')
        else:
            discriminatorModel = gan_discriminator

        return discriminatorModel
