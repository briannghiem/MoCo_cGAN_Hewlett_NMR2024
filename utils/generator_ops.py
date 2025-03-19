import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate, Dropout, MaxPooling3D
from tensorflow.python.keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
# from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Input, Model

class get_generator(object):
    def __init__(self,config):
        config.generator.img_shape = config.data.img_shape
        self.config = config.generator

    def build(self):
        tf.print("Building generator...")
        config = self.config
        i = Input(shape = config.img_shape)
        o = self.level_block(i, config.depth)
        o = Conv3D(config.out_ch, 1, activation=self.config.out_activation)(o)
        return Model(inputs = i, outputs =  o, name='generator')    
   
    def level_block(self,m,depth):
        config = self.config
        n_filters = config.filters[depth]
        
        if depth > 0:
            # Convolutional block
            n = self.conv_block(m, n_filters, do=0)
            
            # Reduce
            m = MaxPooling3D(pool_size=(2,2,2))(n) if config.maxpool else Conv3D(n_filters, 3, strides=2, padding='same')(n)
            
            # Pass to next block
            m = self.level_block(m, depth-1)
            
            # Upconv if UNET
            if config.upconv:
                m = UpSampling3D(size=(2,2,2))(m)
                m = Conv3D(n_filters, 2, activation=config.activation, padding='same')(m)
            else:
                m = Conv3DTranspose(n_filters, 3, strides=2, activation=config.activation, padding='same')(m)
                    
            n = Concatenate()([n, m])
            m = self.conv_block(n, n_filters, do=0)
        else:
            m = self.conv_block(m, n_filters, do=config.dropout)
            
        return m

    def conv_block(self,m, dim, do=0):
        config = self.config
        n = Conv3D(dim, 3, activation=config.activation, padding='same')(m)
        n = BatchNormalization()(n) if config.batchnorm else n
        n = Dropout(rate=do)(n) if do else n
        n = Conv3D(dim, 3, activation=config.activation, padding='same')(n)
        n = BatchNormalization()(n) if config.batchnorm else n
        return Concatenate()([m, n]) if config.residual else n