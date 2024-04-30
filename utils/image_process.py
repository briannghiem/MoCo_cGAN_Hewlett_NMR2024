import numpy as np
import tensorflow as tf
from scipy import interpolate

def combine_coils(image,scale_args,coil_dim=3,CSMsConj=[]):
    '''
        Combines multichannel data
    '''
    
    # Number of channels (for scaling)
    nChan = tf.cast(image.shape[coil_dim],dtype=tf.float32)
    
    # Check for CSMs
    if tf.is_tensor(CSMsConj):
        # Perform coil combination using CSM
        scaleFactor = tf.math.multiply(tf.math.reduce_sum(tf.math.square(tf.math.abs(CSMsConj)),axis=coil_dim,keepdims=True),tf.math.sqrt(nChan))
        numerator = tf.math.reduce_sum(tf.math.multiply(CSMsConj,image),axis=coil_dim,keepdims=True)
        img_combined = tf.math.divide_no_nan(numerator,tf.cast(scaleFactor,dtype=tf.complex64))
    else:
        # Use RSS        
        image = tf.math.abs(image)
        scaleFactor = tf.math.sqrt(nChan)
        numerator = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(image),axis=coil_dim,keepdims=True))
        img_combined = tf.math.divide_no_nan(numerator,scaleFactor)

    # Renormalize
    if not scale_args:
        # Normalize
        img_combined, img_min, img_max = image_normalize(img_combined) 
    elif scale_args == -1:
        # Scale to range [0,1] using present min/max values
        img_min = tf.math.reduce_min(tf.math.abs(img_combined))
        img_max = tf.math.reduce_max(tf.math.abs(img_combined))
        img_combined = image_scale(img_combined,img_min,img_max)
    else:
        # Scale
        img_min = scale_args[0]
        img_max = scale_args[1]
        img_combined = image_scale(img_combined, img_min, img_max)
            
    return img_combined, img_min, img_max

def extract_complex(img_complex,coil_dim=3):
    '''
        Take complex tensor and seperate into real and imaginary components
    '''
    
    real = tf.math.real(img_complex)
    imag = tf.math.imag(img_complex)
    
    return tf.concat([real,imag],axis=coil_dim)

def gen_complex(imageMAG,imagePHASE):
    '''
        Take magnitude and phase components and generate complex tensor
    '''
    
    imageMAG = tf.complex(imageMAG,tf.zeros(tf.shape(imageMAG),dtype=tf.float32)) # A
    imagePHASE = tf.complex(tf.zeros(tf.shape(imagePHASE),dtype=tf.float32),imagePHASE) # iPhi
    imageCOMPLEX = tf.math.multiply(imageMAG,tf.math.exp(imagePHASE))
    
    return imageCOMPLEX

def gen_complexRI(imageCONCAT,coil_dim):
    '''
        Take tensor with concatenated RI components and convert to complex equivalent
    '''
    
    # Conform inputs
    if coil_dim == 3:
       squeezeFlag = True 
       imageCONCAT = tf.expand_dims(imageCONCAT,0)
    elif coil_dim == 4:
       squeezeFlag = False
       
    # Determine number of channels
    tf.shape(imageCONCAT)
    if np.mod(imageCONCAT.shape[4],2):
        n_channels = -1 # Error
    else:
        n_channels = int(imageCONCAT.shape[4]/2)    
    
    imageCOMPLEX = []
    for channel in range(n_channels):
        ch_complex = tf.complex(imageCONCAT[:,:,:,:,channel],imageCONCAT[:,:,:,:,channel + n_channels])
        imageCOMPLEX.append(ch_complex)
        
    imageCOMPLEX = tf.stack(imageCOMPLEX,axis=4)
    
    if squeezeFlag:
        imageCOMPLEX = tf.squeeze(imageCOMPLEX,axis = 0)
        
    return imageCOMPLEX

def get_percentile(imageData,queries,method='NEAREST'):
    
    imageData = tf.reshape(imageData,[-1])
    imageData = tf.sort(imageData)
    result = []
    for query in queries:
        elemQuery = tf.cast(tf.math.round(imageData.shape[0]*query/100),dtype=tf.int64)
        result.append(imageData[elemQuery])
    
    return result

def image_normalize(image):
    '''
        Normalizes a single image
    '''
    
    if image.dtype == tf.complex64:
        imageMAG = tf.math.abs(image)
    else:
        imageMAG = image
    
    image_min, image_max = get_percentile(imageMAG,tf.constant([0.05,99.95],dtype=tf.float32))
    image_normalized = image_scale(image, image_min, image_max)
    
    return image_normalized, image_min, image_max

def image_pad(img):
    scale_factor = 4
    img_plot = np.zeros((np.multiply(scale_factor,img.shape[0]),
                        np.multiply(scale_factor,img.shape[1]),
                        img.shape[2]))
    for slice_ax in range(img.shape[2]):
        y = np.arange(0,img.shape[0],1)
        x = np.arange(0,img.shape[1],1)
        f = interpolate.interp2d(x, y, img[:,:,slice_ax], kind='cubic')
        
        # Interpolate
        ynew = np.arange(0,img.shape[0],1/scale_factor)
        xnew = np.arange(0,img.shape[1],1/scale_factor)
        img_plot[:,:,slice_ax] = f(xnew,ynew)

    return img_plot

def image_scale(image,dataMin,dataMax):
    
    if image.dtype == tf.complex64:
        imageMAG = tf.math.abs(image)
    else:
        imageMAG = image
    
    # Scale to range [0,1]
    image_scaled = tf.math.divide(tf.math.subtract(imageMAG,dataMin),dataMax - dataMin)
    image_scaled = tf.where(tf.math.less(image_scaled,tf.constant(0,dtype=tf.float32)),tf.constant(0,dtype=tf.float32),image_scaled)
    image_scaled = tf.where(tf.math.greater(image_scaled,tf.constant(1,dtype=tf.float32)),tf.constant(1,dtype=tf.float32),image_scaled)
    
    if image.dtype == tf.complex64:
        image_scaled = gen_complex(image_scaled,tf.math.angle(image))
        
    return image_scaled