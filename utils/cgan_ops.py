"""
Define the model class for the cGAN network
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from utils.discriminator_ops import get_discriminator
from utils.generator_ops import get_generator
from utils.image_process import combine_coils, gen_complexRI, image_pad

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

class cGAN(object):
    #---------------------------------------------------------------
    def __init__(self,config):
        if config.data.channel_opt == 'ALL' and config.data.complex_flag == True:
            config.data.csm_flag = True
        else:
            config.data.csm_flag = False
        #
        self.config = config
        #
        with config.strategy.scope():
            # Set up generator, discriminator
            self.generator = get_generator(config).build()            
            self.discriminator = get_discriminator(config).build()
            self.config.discriminator.optimizer = Adam(learning_rate=config.training.learning_rate, epsilon=1e-07, decay=0)  
            #
            # Set up the cGAN
            self.discriminator.trainable = False
            generator_input = Input(shape=config.data.img_shape, name="CGAN_input")        
            generated_image = self.generator(generator_input,training=False)            
            #
            # Complex multichannel discriminator takes CSMs as additional input for coil combination purposes
            if config.data.csm_flag:
                CSMs = Input(shape=self.config.discriminator.csm_shape,dtype=tf.complex64,name="CSMs")
                disc_output = self.discriminator([generated_image,CSMs],training=False)
                self.cgan = Model(inputs=[generator_input,CSMs], outputs=[generated_image, disc_output], name="CGAN")
            else:
                disc_output = self.discriminator(generated_image,training=False)
                self.cgan = Model(inputs=[generator_input], outputs=[generated_image, disc_output], name="CGAN")
            self.config.cGAN.optimizer = Adam(learning_rate=config.training.learning_rate, epsilon=1e-07, decay=0)            
            #
            # Turn discriminator training back on
            self.discriminator.trainable = True
            #
            if config.summary:
                self.summary()
    #
    def summary(self):
        def summary_save(text):
            with open(os.path.join(self.config.summary_dir,'model_summary.txt'),'a+') as f:
                print(text,file=f)
        #
        open(os.path.join(self.config.summary_dir,'model_summary.txt'),'w').close()
        #
        # Generate summarys
        self.generator.summary(line_length = 125,
                               positions=[.40, .65, .75, 1.],
                               print_fn=summary_save)
        #
        self.discriminator.summary(line_length = 125,
                                   positions=[.40, .65, .75, 1.],
                                   print_fn=summary_save)
        #
        self.discriminator.trainable = False
        self.cgan.summary(line_length = 125,
                          positions=[.40, .65, .75, 1.],
                          print_fn=summary_save)
        self.discriminator.trainable = True
    #
    #---------------------------------------------------------------
    def train(self,train_data, val_data):
        config = self.config
        #
        # Configure model for training
        epoch_start, ckpt, save_manager = self.configure()        
        tf.print('Training networks ...')
        for epoch in range(epoch_start,config.training.num_epochs): # loop through epochs
            #
            #------------------------------
            #Training & Validation
            with config.strategy.scope():
                tf.print('Epoch {} - Training'.format(epoch + 1)) 
                t_data = self.data_reset(train_data)        
                t_loss = self.distributed_train_step(t_data)  
                #
                tf.print('Epoch {} - Validation'.format(epoch + 1)) 
                v_data = self.data_reset(val_data)    
                v_loss = self.distributed_val_step(v_data)        
            #
            # Print losses
            for ix, loss in enumerate(config.loss_names):
                self.losses[loss]['training'].append(t_loss[ix].numpy())
                self.losses[loss]['validation'].append(v_loss[ix].numpy())
            #
            # Save checkpoint
            ckpt.n_epoch.assign_add(1)
            save_manager.save()
            tf.print('Checkpoint saved')    
            # Save losses
            spath_loss_temp = os.path.join(self.config.save_dir,'checkpoints','loss.npy')
            np.save(spath_loss_temp, self.losses)
    #
    def configure(self):
        self.reset_states()
        #
        epoch_start = tf.Variable(0)
        self.losses = {self.config.loss_names[0]: {'training': [],'validation': []},
                       self.config.loss_names[1]: {'training': [],'validation': []},
                       self.config.loss_names[2]: {'training': [],'validation': []},
                       self.config.loss_names[3]: {'training': [],'validation': []},
                       } #{'Discriminator loss', 'cGAN total loss', 'cGAN generator loss', 'cGAN discriminator loss'}
        with self.config.strategy.scope():        
            ckpt = tf.train.Checkpoint(n_epoch = epoch_start,
                                       discriminator = self.discriminator,
                                       generator = self.generator,
                                       cgan = self.cgan,
                                       discriminator_opt = self.config.discriminator.optimizer,
                                       cgan_opt = self.config.cGAN.optimizer)
            #
            ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                      os.path.join(self.config.save_dir,'checkpoints'),
                                                      max_to_keep=self.config.training.ckpts_to_keep)        
            #
            if self.config.load.opt:
                if self.config.load.checkpoint == 'latest':
                    checkpoint_load = ckpt_manager.latest_checkpoint
                else:
                    checkpoint_load = os.path.join(self.config.save_dir,'checkpoints','ckpt-' + self.config.load.checkpoint)
                #
                tf.print('Loading saved checkpoint from ' + checkpoint_load)
                ckpt.restore(checkpoint_load)
            else:
                tf.print('Training from beginning...')
                ckpt_manager.save()
                tf.print('Initial state saved as ' + ckpt_manager.latest_checkpoint)
        #
        return int(epoch_start), ckpt, ckpt_manager        
    #
    def reset_states(self):
        # Reset models
        with self.config.strategy.scope():
            self.generator.reset_states()
            self.discriminator.reset_states()
            self.cgan.reset_states()           
    #
    def data_reset(self,data):
        def data_generator():
            for sample in tf.range(self.data.count): #iterating through all samples
                # Load generator data
                sample_idx = self.data.idxs[sample]
                sample_file = self.data.file_identifier[sample_idx]
                sample_channel = self.data.channel_identifier[sample_idx]
                #
                G_input, G_output, CSMs = self.data.load(sample_file,sample_channel)
                if not self.config.data.csm_flag:
                    CSMs = tf.zeros([0])
                #
                # Generate corresponding discriminator data
                batch = tf.math.floor(tf.math.divide(sample,self.data.batch_size))
                D_input, D_output = self.get_discriminator_batch(G_output,G_input,batch+1)
                GD_output = tf.constant([0,1],dtype=tf.int32) # real images
                #
                yield G_input, G_output, GD_output, D_input, D_output, CSMs
        #
        data.reset()
        self.data = data
        #
        if self.config.data.csm_flag:
            CSMshape = tf.TensorShape(self.config.discriminator.csm_shape)
        else:
            CSMshape = tf.TensorShape([0])
        #
        dataset = tf.data.Dataset.from_generator(data_generator,
                                                 (tf.float32, tf.float32,tf.float32, tf.float32, tf.float32, tf.complex64),
                                                 output_shapes=(tf.TensorShape(self.config.data.img_shape),
                                                                tf.TensorShape(self.config.data.img_shape),
                                                                tf.TensorShape([2]),
                                                                tf.TensorShape(self.config.data.img_shape),
                                                                tf.TensorShape([2]),
                                                                CSMshape))
        #
        dataset = dataset.batch(data.batch_size,drop_remainder=True)
        dataset = self.config.strategy.experimental_distribute_dataset(dataset)
        #
        return dataset
    #
    #---------------------------------------------------------------
    #HELPER FUNCTIONS FOR TRAINING
    @tf.function
    def distributed_train_step(self,dataset): #distributed across multiple devices
        batch = 0
        total_losses = tf.zeros([4],tf.float32)
        for batch_data in dataset:
            tf.print('Training Batch Number',batch + 1)
            #
            d_loss_per_rep, t_loss_per_rep, g_loss_per_rep, gd_loss_per_rep = self.config.strategy.run(self.train_step,args=batch_data)
            #
            d_loss_combined = self.config.strategy.reduce(tf.distribute.ReduceOp.SUM,d_loss_per_rep,axis=None)
            t_loss_combined = self.config.strategy.reduce(tf.distribute.ReduceOp.SUM,t_loss_per_rep,axis=None)
            g_loss_combined = self.config.strategy.reduce(tf.distribute.ReduceOp.SUM,g_loss_per_rep,axis=None)
            gd_loss_combined = self.config.strategy.reduce(tf.distribute.ReduceOp.SUM,gd_loss_per_rep,axis=None)
            #
            total_losses += tf.stack([d_loss_combined,t_loss_combined,g_loss_combined,gd_loss_combined])
            #
            batch += 1
        #
        return total_losses / tf.cast(batch, dtype=tf.float32)
    #
    def train_step(self,g_input,g_output,gd_output,d_input,d_output,csms=[]):        
        d_loss = self.train_discriminator(d_input,d_output,csms)        
        t_loss, g_loss, gd_loss = self.train_cgan(g_input, g_output, gd_output,csms)
        #
        return d_loss, t_loss, g_loss, gd_loss
    #
    def train_discriminator(self,d_input,d_output,csms=[]):        
        with tf.GradientTape() as tape:
            if self.config.data.csm_flag:
                d_prediction = self.discriminator([d_input,csms], training=True)
            else:
                d_prediction = self.discriminator(d_input, training=True)
            #
            d_loss = self.compute_discriminator_loss(d_output, d_prediction)
        #
        gradients = tape.gradient(d_loss,self.discriminator.trainable_variables)
        self.config.discriminator.optimizer.apply_gradients(zip(gradients,self.discriminator.trainable_variables))
        #
        return d_loss
    #
    def train_cgan(self,g_input,g_output,gd_output,csms=[]):
        with tf.GradientTape() as tape:
            if self.config.data.csm_flag:
                g_prediction, gd_prediction = self.cgan([g_input,csms], training=True)
            else:
                g_prediction, gd_prediction = self.cgan(g_input, training=True)
            #
            t_loss, g_loss, gd_loss = self.compute_cgan_loss_train(g_output, g_prediction, gd_output, gd_prediction)
        #
        gradients = tape.gradient(t_loss,self.cgan.trainable_variables)
        self.config.cGAN.optimizer.apply_gradients(zip(gradients,self.cgan.trainable_variables))
        #
        return t_loss, g_loss, gd_loss
    #
    def compute_discriminator_loss(self,ground_truth,prediction):
        loss_per_sample = self.config.discriminator.loss_func(ground_truth,prediction)
        combined_loss = tf.nn.compute_average_loss(loss_per_sample,
                                                   global_batch_size=self.config.training.batch_size)
        #
        return combined_loss
    #
    def compute_cgan_loss_train(self,g_ground_truth, g_prediction, gd_ground_truth, gd_prediction):
        g_ground_truth = tf.expand_dims(g_ground_truth,axis=-1)
        g_prediction = tf.expand_dims(g_prediction,axis=-1)
        g_ae_loss_per_sample = self.config.cGAN.loss_func_g(g_ground_truth,g_prediction) #compute MAE loss
        g_mae_loss_per_sample = tf.math.reduce_mean(g_ae_loss_per_sample, axis=4)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=3)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=2)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=1)
        g_loss_combined = tf.nn.compute_average_loss(g_mae_loss_per_sample,
                                                     global_batch_size=self.config.training.batch_size)
        #
        gd_loss_per_sample = self.config.cGAN.loss_func_d(gd_ground_truth,gd_prediction) #compute Binary Cross-Entropy
        gd_loss_combined = tf.nn.compute_average_loss(gd_loss_per_sample,
                                                      global_batch_size=self.config.training.batch_size)
        #
        weight_g = self.config.cGAN.loss_weights[0]
        weight_d = self.config.cGAN.loss_weights[1]
        t_loss_combined = tf.math.add(weight_g*g_loss_combined,weight_d*gd_loss_combined)
        #
        return t_loss_combined, g_loss_combined, gd_loss_combined
    #
    def get_discriminator_batch(self,images,images_motion,batch):
        mod = tf.math.floormod(tf.cast(batch,dtype=tf.float32),tf.constant(2,dtype=tf.float32))
        if mod == tf.constant(0,dtype=tf.float32):
            # Generate 'fake' images
            X = self.generator(tf.expand_dims(images_motion,0),training=False)
            X = tf.squeeze(X,axis=0)
            #
            # Define output (1x2 vector for the result)
            y = tf.constant([1,0],dtype=tf.int32) # fake images
        #
        elif mod == tf.constant(1,dtype=tf.float32):
            # Input 'real' (ground truth) images
            X = images
            #
            # Define output
            y = tf.constant([0,1],dtype=tf.int32) # real images
        #
        return X, y
    #
    #---------------------------------------------------------------
    #HELPER FUNCTIONS FOR VALIDATION
    def distributed_val_step(self,dataset):
        batch = 0
        total_losses = tf.zeros([4],tf.float32)
        for batch_data in dataset:
            tf.print('Validation Batch Number',batch + 1)
            #
            d_loss_per_rep, t_loss_per_rep, g_loss_per_rep, gd_loss_per_rep = self.config.strategy.run(self.val_step,args=batch_data)
            #
            d_loss_combined = None
            t_loss_combined = None
            g_loss_combined = self.config.strategy.reduce(tf.distribute.ReduceOp.SUM,g_loss_per_rep,axis=None)
            gd_loss_combined = None
            #
            total_losses += tf.stack([d_loss_combined,t_loss_combined,g_loss_combined,gd_loss_combined])
            #
            batch += 1
        #
        return total_losses / tf.cast(batch, dtype=tf.float32)
    #
    def val_step(self,g_input,g_output,gd_output,d_input,d_output,csms=[]):        
        d_loss = None
        t_loss, g_loss, gd_loss = self.val_generator(g_input, g_output, gd_output,csms)
        #
        return d_loss, t_loss, g_loss, gd_loss #only need g_loss; the others pertain to discriminator, which is not evaluated at validation step 
    #
    def val_generator(self,g_input,g_output,gd_output,csms=[]):
        with tf.GradientTape() as tape:
            if self.config.data.csm_flag:
                g_prediction, gd_prediction = self.cgan([g_input,csms], training=False)
            else:
                g_prediction, gd_prediction = self.cgan(g_input, training=False) #will ignore discriminator output during validation step
                # g_prediction = self.generator(tf.expand_dims(img_motion, axis=0), training=False)
            #
            t_loss, g_loss, gd_loss = self.compute_cgan_loss_val(g_output, g_prediction, gd_output, gd_prediction)
        #
        return t_loss, g_loss, gd_loss
    #
    def compute_cgan_loss_val(self,g_ground_truth, g_prediction, gd_ground_truth, gd_prediction):
        #Computing only MAE loss during validation, since only evaluating the generator
        g_ground_truth = tf.expand_dims(g_ground_truth,axis=-1)
        g_prediction = tf.expand_dims(g_prediction,axis=-1)
        g_ae_loss_per_sample = self.config.cGAN.loss_func_g(g_ground_truth,g_prediction) #compute MAE loss
        g_mae_loss_per_sample = tf.math.reduce_mean(g_ae_loss_per_sample, axis=4)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=3)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=2)
        g_mae_loss_per_sample = tf.math.reduce_mean(g_mae_loss_per_sample, axis=1)
        g_loss_combined = tf.nn.compute_average_loss(g_mae_loss_per_sample,
                                                     global_batch_size=self.config.training.batch_size)
        #
        gd_loss_combined = None
        t_loss_combined = None
        #
        return t_loss_combined, g_loss_combined, gd_loss_combined
    
    # #---------------------------------------------------------------
    # def sample_output(self,dataset,title,epoch):
    #     # Get random sample
    #     ix = np.random.choice(dataset.idxs)
    #     file = dataset.file_identifier[ix]
    #     channel = dataset.channel_identifier[ix]
        
    #     # Load data
    #     img_motion, img_true, CSMs = dataset.load(file,channel)
        
    #     # Get predicted image
    #     img_corrected = self.generator(tf.expand_dims(img_motion, axis=0), training=False)
    #     img_corrected = tf.squeeze(img_corrected,axis=0)
        
    #     if dataset.complex_flag:
    #         # Convert to complex representation
    #         img_true = gen_complexRI(img_true,coil_dim=3)
    #         img_motion = gen_complexRI(img_motion,coil_dim=3)
    #         img_corrected = gen_complexRI(img_corrected,coil_dim=3)
        
    #     # Perform coil combination for multichannel network, cropping unnecessary channels
    #     if self.config.data.channel_opt == 'ALL':
    #         # Crop unnecessary channels
    #         if dataset.complex_flag:
    #             CSMs = CSMs[:,:,:,0:dataset.n_channels[ix]]
    #         else:
    #             CSMs = []
    #         img_true,_,_ = combine_coils(img_true[:,:,:,0:dataset.n_channels[ix]],-1,coil_dim=3,CSMsConj=CSMs)
    #         img_motion,_,_ = combine_coils(img_motion[:,:,:,0:dataset.n_channels[ix]],-1,coil_dim=3,CSMsConj=CSMs)
    #         img_corrected,_,_ = combine_coils(img_corrected[:,:,:,0:dataset.n_channels[ix]],-1,coil_dim=3,CSMsConj=CSMs)            
            
    #     # Plot
    #     self.plot_sample(img_true,img_motion,img_corrected,title,epoch,im_range=[0,1])               

    # def plot_sample(self,img_true,img_motion,img_corrected,title,epoch,im_range=[-1.5,3.5],er_range=[0,1],slc=3):
        
    #     # For complex data take magnitude
    #     if img_motion.dtype == tf.complex64:
    #         if tf.is_tensor(img_true):
    #             img_true = tf.math.abs(img_true)
    #         img_motion = tf.math.abs(img_motion)
    #         img_corrected = tf.math.abs(img_corrected)        
        
    #     # Set reference for difference images
    #     if tf.is_tensor(img_true):
    #         img_ref = img_true
    #     else:
    #         img_ref = img_motion        
        
    #     # Get channel to plot
    #     img_motion = image_pad(np.array(img_motion)[:,:,:,0])
    #     img_corrected = image_pad(np.array(img_corrected)[:,:,:,0])
    #     img_ref = image_pad(np.array(img_ref)[:,:,:,0])
    #     if tf.is_tensor(img_true):
    #         img_true = image_pad(np.array(img_true)[:,:,:,0])
    #     else:
    #         img_true = np.zeros(img_motion.shape)        
        
    #     # Plot
    #     fig, ax = plt.subplots(2, 3, figsize = (20,15), dpi = 300)
    #     ax[0,0].imshow(img_true[:,:,slc], cmap='gray', vmin=im_range[0], vmax=im_range[1])
    #     ax[0,0].axis('off')
    #     ax[0,0].set_title(title + ' Epoch ' + str(epoch))
    #     ax[1,0].axis('off')
    #     ax[0,1].imshow(img_corrected[:,:,slc], cmap='gray', vmin=im_range[0], vmax=im_range[1])
    #     ax[0,1].axis('off')
    #     ax[1,1].imshow(np.abs(img_corrected[:,:,slc] - img_ref[:,:,slc]), cmap='gray', vmin=er_range[0], vmax=er_range[1])
    #     ax[1,1].axis('off')
    #     ax[0,1].set_title('Corrected')
    #     ax[0,2].imshow(img_motion[:,:,slc], cmap='gray', vmin=im_range[0], vmax=im_range[1])
    #     ax[0,2].axis('off')
    #     ax[0,2].set_title('Original')
    #     ax[1,2].imshow(np.abs(img_motion[:,:,slc] - img_ref[:,:,slc]), cmap='gray', vmin=er_range[0], vmax=er_range[1])
    #     ax[1,2].axis('off')        
        
    #     plt.show()
