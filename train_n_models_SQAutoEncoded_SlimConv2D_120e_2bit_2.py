# MODEL DETAILS ---------------------------------------------------------------------


# Gaussian noise parameters
NOISE_MU = 0.0
NOISE_SIGMA = 120.0 # e-

# Training name
training_name = f'SQAutoEncoder_SlimConv2D_{str(int(NOISE_SIGMA))}e_2bit_2'

# Precision of input data
N_BITS = 2

# Number of trainings to run
import argparse
parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('-n','--num', help='Number of trainings to run', required=True)
parser.add_argument('--init_SQDAE', help='path to SQAutoEncoder to initialize from', default = None)

args = vars(parser.parse_args())

n_tries = int(args['num'])
print("Running " + str(n_tries) + " trainings with " + str(NOISE_SIGMA) + "e noise")

# IMPORTS ---------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import random
import time

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence
from keras.layers import Conv2D, MaxPooling2D
from qkeras import *

from keras.utils import Sequence
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, Callback
from keras.losses import CategoricalCrossentropy
import csv

from loss import *
from SoftQuantizeLayer import SoftQuantizeLayer
from AnnealingScheduler import AnnealingScheduler
from DAE_DataGenerator import OptimizedDataGenerator as DAE_OptDG
from OptimizedDataGenerator_v2 import OptimizedDataGenerator
from dataset_utils import add_noise, quantize_manual
from plotting import *
from models import *


pi = 3.14159265359
maxval=1e9
minval=1e-9

# TRAINING DATA ---------------------------------------------------------------------

dataset_base_dir = "/uscms/home/bweiss/nobackup/smart-pixels/"
tfrecords_base_dir = "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords"

dataset_dir_train = os.path.join(dataset_base_dir, "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets", 'train_contained/')
dataset_dir_val = os.path.join(dataset_base_dir, "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets", 'test_contained/')

SQDAE_tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train",'SQDAE_3srCI_'+str(int(NOISE_SIGMA))+'eN_catgoricalOut_Target_0_30_845') 
SQDAE_tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val",'SQDAE_3srCI_'+str(int(NOISE_SIGMA))+'eN_catgoricalOut_Target_0_30_845')

REG_tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train",'3srCI_16x16_'+str(int(NOISE_SIGMA))+'eN_raw_slim')
REG_tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val",'3srCI_16x16_'+str(int(NOISE_SIGMA))+'eN_raw_slim')

seeds = []

with open('log_'+training_name+'.txt','a') as f:
        f.write("====================="+ training_name + "======================"+ "\n")

# BEGIN LOOP ---------------------------------------------------------------------
for i in range(n_tries):
    SQDAE_start_time = time.perf_counter()

    print("Running training " + str(i) + " of model " + training_name)   
    with open('log_'+training_name+'.txt','a') as f:
        f.write("Running training " + str(i) + " of " + str(n_tries) + " model " + training_name + "\n")
    
    seed = random.randint(0, 1000)
    while True:
        if seed in seeds:
            seed = random.randint(0, 1000)
        else:
            break

    print("Seed: ", seed)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Seed: ' + str(seed) + "\n")
    
    # Loading pre-generated SQAutoEncoder TFRecords
    validation_generator = DAE_OptDG(
        load_from_tfrecords_dir= SQDAE_tfrecords_dir_val,
        shuffle=True,
        seed=seed,
        quantize=False,
        bitwise_categorical = True,
    )

    training_generator = DAE_OptDG(
        load_from_tfrecords_dir = SQDAE_tfrecords_dir_train,
        shuffle=True,
        seed=seed,
        quantize=False,
        bitwise_categorical = True,
    )
    with open('log_'+training_name+'.txt','a') as f:
        f.write('SQDAE TFR_val directory: ' + SQDAE_tfrecords_dir_val + "\n")
        f.write('SQDAE TFR_train directory: ' + SQDAE_tfrecords_dir_train + "\n")

    # Initial thresholds from transformer
    initial_thresholds = [250,400,800]
    
    print("Initial thresholds: ", initial_thresholds)
    with open('log_'+training_name+'.txt','a') as f:
        f.write("Initial thresholds: " + str(initial_thresholds) + "\n")
    
    
    model=CreateSQAutoencoder_Conv2D((16,16,2), n_filters=5, pool_size=3,
                            decoder_filters = 128, initial_thresholds=initial_thresholds, 
                                         OutBits = 2, conv_kernel_size = 3)
    
    #UNCOMMENT to Initialize from Weights
    # weights_path = './SQ_DAE/model-20014e96-SQDAE_Conv2D_Categorical_2bit_0eN_Target_0_30_856_V2-checkpoints/'
    # base_dir = weights_path
    # checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    # best_weights = max(checkpoint_files, key=os.path.getmtime)
    # print(f"Loading model from {best_weights}")
    # model.load_weights(best_weights)
    
    
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-3, clipnorm=1.0),
        loss = CategoricalCrossentropy(reduction = 'sum', from_logits=True
                                      )
    )

    #Training setup
    fingerprint = '%08x' % random.randrange(16**8)

    print("Fingerprint: ", fingerprint)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Fingerprint: ' + str(fingerprint) + "\n")
    home_dir = f'/uscms/home/bweiss/nobackup/smart-pixels/'
    base_dir = home_dir + f'SQAutoEncoder_models/model-{fingerprint}-{training_name}-checkpoints'
    # checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    checkpoint_filepath = os.path.join(base_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')
    
    os.makedirs(home_dir+"SQAutoEncoder_models", exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    # os.makedirs(checkpoints_dir, exist_ok=True) 
    
    early_stopping_patience = 1200

    class CustomModelCheckpoint(ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            checkpoints = [f for f in os.listdir(base_dir) if f.startswith('weights')]
            if len(checkpoints) > 1:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir,x)))
                for checkpoint in checkpoints[:-1]:
                    os.remove(os.path.join(base_dir, checkpoint))
    
    class SoftQuantizeLoggerCallback(Callback):
        def __init__(self, log_filepath, layer_name="soft_quantizer_output"):
            super().__init__()
            self.log_filepath = log_filepath
            self.layer_name = layer_name
            self.header_written = False
    
        def on_train_begin(self, logs=None):
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
    
        def on_epoch_end(self, epoch, logs=None):
            try:
                layer = self.model.get_layer(self.layer_name)
                if not hasattr(layer, 'n_bits'):
                     print(f"\nWarning: Layer '{self.layer_name}' is not a SoftQuantizeLayer. Skipping logging.")
                     return
            except ValueError:
                print(f"\nWarning: Layer '{self.layer_name}' not found in the model. Skipping logging.")
                return
    
            if not self.header_written:
                num_levels = layer.num_levels
                num_thresholds = num_levels - 1
                
                header = ['epoch', 'k']
                # header.extend([f'level_{i}' for i in range(num_levels)])
                header.extend([f'threshold_{i}' for i in range(num_thresholds)])
    
                # header.append('raw_first_level')
                # header.extend([f'raw_log_level_delta_{i}' for i in range(num_levels - 1)])
                # header.append('raw_first_threshold')
                # if hasattr(layer, 'log_threshold_deltas'):
                #     header.extend([f'raw_log_threshold_delta_{i}' for i in range(num_thresholds - 1)])
    
                with open(self.log_filepath, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                self.header_written = True
    
            k_val = layer.k.numpy().item()
            # levels = layer.levels.numpy().tolist()
            thresholds = layer.thresholds.numpy().tolist()
            
            # first_level = layer.first_level.numpy().item()
            # log_level_deltas = layer.log_level_deltas.numpy().tolist()
            # first_threshold = layer.first_threshold.numpy().item()
            
            row_data = [epoch, k_val]
            # row_data.extend(levels)
            row_data.extend(thresholds)
            # row_data.append(first_level)
            # row_data.extend(log_level_deltas)
            # row_data.append(first_threshold)
            
            # if hasattr(layer, 'log_threshold_deltas'):
            #     log_threshold_deltas = layer.log_threshold_deltas.numpy().tolist()
            #     row_data.extend(log_threshold_deltas)
            
            with open(self.log_filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
    
    
    csv_logger = CSVLogger(f'{base_dir}/training_log.csv', append=True)
    scheduler_callback = AnnealingScheduler(
        schedule='cosine',  
        target_layer_name='soft_quantizer_output', 
        initial_k=1.0,
        final_k=400.0, 
        verbose=1      
    )
    
    quantizer_logger = SoftQuantizeLoggerCallback(
        log_filepath=f"{base_dir}/soft_quantizer_state_log.csv", # New, more descriptive filename
        layer_name="soft_quantizer_output"
    )
    
    es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
    
    mcp = CustomModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    #Train SQAutoEncoder
    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        callbacks=[es, mcp, csv_logger, scheduler_callback, quantizer_logger],
        epochs=4000, #5000
        shuffle=True,
        verbose=1
    )
    
    del training_generator
    del history

    #Save predictions
    print("Saving SQAutoEncoder predictions")
    pred_i = 0
    for x, y in validation_generator:
        df = construct_df_SQAutoEncoder(model, x, y)
        df.to_parquet(f'{base_dir}/SQAutoEncoder_predictions{pred_i}.parquet')
        pred_i+=1
    
    print("Cleaning up from SQAutoEncoder training")
    del validation_generator
    del model
    del pred_i
    
    SQDAE_end_time = time.perf_counter()
    print("SQAutoEncoder Training time: ", SQDAE_end_time - SQDAE_start_time)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('SQAutoEncoder Training time: ' + str(SQDAE_end_time - SQDAE_start_time) + "\n")
    print('Training Slim regression network from SQEncoder')
        
    '''============================================================================='''
    
    REG_start_time = time.perf_counter()
    #Construct a Slim Regression Network with a SQAutoEncoded Conv network
    # Loading pre-generated regression TFRecords
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= REG_tfrecords_dir_val,
        shuffle=True,
        seed=seed,
        quantize=False,
    )

    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = REG_tfrecords_dir_train,
        shuffle=True,
        seed=seed,
        quantize=False,
    )
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Regression TFR_val directory: ' + REG_tfrecords_dir_val + "\n")
        f.write('Regression TFR_train directory: ' + REG_tfrecords_dir_train + "\n")

    model = CreateSQAE_SlimConv2DReg_fromweights((16,16,2), base_dir, FEBits = 2, 
                                            OutBits = 2, n_filters=5, pool_size=3, 
                                            conv_kernel_size=3, decoder_filters = 128)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss=custom_sse_loss
    )

    fingerprint = '%08x' % random.randrange(16**8)

    print("regression Fingerprint: ", fingerprint)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Regression Fingerprint: ' + str(fingerprint) + "\n")
    
    reg_base_dir = home_dir + f'SQAutoEncoder_regression_models/model-{fingerprint}-{training_name}-checkpoints'
    os.makedirs(home_dir+"SQAutoEncoder_regression_models", exist_ok=True)
    os.makedirs(reg_base_dir, exist_ok=True)
    # checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    # os.makedirs(checkpoints_dir, exist_ok=True) 

    class CustomModelCheckpoint(ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            checkpoints = [f for f in os.listdir(reg_base_dir) if f.startswith('weights')]
            if len(checkpoints) > 1:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(reg_base_dir,x)))
                for checkpoint in checkpoints[:-1]:
                    os.remove(os.path.join(reg_base_dir, checkpoint))
                    
    checkpoint_filepath = os.path.join(reg_base_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')

    early_stopping_patience = 1200
    es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

    mcp = CustomModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    csv_logger = CSVLogger(f'{reg_base_dir}/training_log.csv', append=True)
    
    #Train SQAutoEncoded SlimConv2D
    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        callbacks=[es, mcp, csv_logger],
        epochs=4000, #5000
        shuffle=False,
        verbose=1
    )

    #Save predictions
    sf = validation_generator.labels_scale

    scale_factor = {}
    scale_factor['x'] = sf[0]
    scale_factor['y'] = sf[1]
    # scale_factor['cotA'] = sf[2]
    scale_factor['cotB'] = sf[2]

    df = construct_df_slim(model, validation_generator, scale_factor)
    df.to_parquet(f"{reg_base_dir}/residuals.parquet")
    print("Cleaning up from regression training")
    del validation_generator
    del training_generator
    del model
    del history
    
    REG_end_time = time.perf_counter()
    with open('log_'+training_name+'.txt','a') as f:
        f.write("SQAutoEncoder regression Training time: " + str(REG_end_time - REG_start_time) + "\n")

# END LOOP ---------------------------------------------------------------------

print("Finished!")
