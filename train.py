'''
    Main file for training a cGAN for motion correction in MRI.
'''

from utils.cgan_ops import cGAN
from utils.dataset_prep import prepare_train_data
from utils.model_config import Config

#-------------------------------------------------------------------------------
# Limit Tensorflow GPU usage
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#limit GPU memory usage
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


#-------------------------------------------------------------------------------
# Options
model_type = 'cGAN_complex' # 'cGAN_complex' (performs motion correction on complex coil-combined data)
                                          # 'cGAN_singlechannel_complex' (performes motion correction on each complex channel independantly)
                                          # 'cGAN_multichannel_complex' (performs motion correction on all complex channels simultaneously)
                                          # Each variation can also be trained to perform correction on magnitude data by removing '_complex'

# Configure settings
if model_type[0:17] == 'cGAN_multichannel':
    config = Config(model_type,filters=[2048,1024,512,256])
else:
    config = Config(model_type)
config.load.opt = 0 # load saved models (turn off to train from scratch)

config.data.dir = r'/home/nghiemb/RMC_repos/MoCo_cGAN/data/training_dataset/slices'

# Set up datasets
train_data = prepare_train_data(config,'training')
validation_data = prepare_train_data(config,'validation')

# Configure model
model = cGAN(config)

if config.load.opt: #i.e., if loading pre-trained model
    if model_type ==  'cGAN_complex':
        config.load.checkpoint = '73'
    elif model_type == 'cGAN_singlechannel_complex':
        config.load.checkpoint = '64'
    elif model_type == 'cGAN_multichannel_complex':
        config.load.checkpoint = '47'  
    config.training.num_epochs = int(config.load.checkpoint) - 1 # skip training and stick with saved model

# Train
model.train(train_data=train_data)

# Sample output
model.sample_output(validation_data,'Validation Example',config.training.num_epochs)