import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.math import *
# import numpy as np


# custom loss function
def custom_loss(y, p_base, minval=1e-9, maxval=1e9, scale = 512):
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    
    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood,minval,maxval)

    NLL = -1*tf.math.log(likelihood)

    return tf.keras.backend.sum(NLL) 

def custom_sse_loss(y, p_base, minval=1e-9, maxval=1e9):

    # truth values
    x_true = y[:,0]
    y_true = y[:,1]
    cotB_true = y[:,2]

    # predictions
    p = p_base
    x_pred = p[:,0]
    y_pred = p[:,1]
    cotB_pred = p[:,2]

    sse_x = tf.reduce_sum(tf.square(x_true - x_pred))
    sse_y = tf.reduce_sum(tf.square(y_true - y_pred))
    sse_cotB = tf.reduce_sum(tf.square(cotB_true - cotB_pred))

    return (sse_x + sse_y + sse_cotB) / 3.0

def log_scale(x):
    non_zeros = tf.cast(tf.not_equal(x, 0.0), tf.bool)
    log_x = tf.math.sign(x) * tf.math.log(tf.math.abs(x)) / tf.math.log(2.0)
    return tf.where(non_zeros, log_x, x)
        
def asymmetric_scaling(x, pos_scale=1.0, neg_scale=1.0):
    """
    Scales positive and negative values differently.

    Args:
        x: A TensorFlow tensor (input data).
        positive_scale: Scaling factor for positive values.
        negative_scale: Scaling factor for negative values.
    Returns:
        A tensor with asymmetrically scaled values.
    """
    # Create masks for positive and negative values
    pos = tf.cast(x >= 0, tf.float32)
    neg = tf.cast(x < 0, tf.float32)

    # Apply different scaling factors
    scaled_positive = x * pos / pos_scale
    scaled_negative = x * neg / neg_scale

    # Combine the scaled results
    return scaled_positive + scaled_negative
    
def CustomSSIM(logxrange, log_standardize):
    logmax=max(logxrange)
    logmin=min(logxrange)
    def SSIMLoss(y_true, y_pred):
        # tf.print("y_pred shape:", y_pred.shape, "y_pred values:", y_pred)
        # tf.print("y_true shape:", y_true.shape,"y_true values:", y_true)
        # tf.print('pred max,min', tf.reduce_max(y_pred), tf.reduce_min(y_pred))
        if log_standardize:
            # print(y_true.shape, y_true.dtype, y_true)
            y_target = log_scale(y_true)
            # tf.print("y_target shape:", y_target.shape,"y_target values:", y_target)
            # tf.print('log max,min', tf.reduce_max(y_target), tf.reduce_min(y_target))
            y_target = asymmetric_scaling(y_target, pos_scale = logmax, neg_scale = abs(logmin))
            # tf.print('scaled log max,min', tf.reduce_max(y_target), tf.reduce_min(y_target))

        # max_val = max(max(y_true), max(y_pred))
        # Calculate SSIM for each image in the batch
        ssim_value = tf.image.ssim(y_target, y_pred, max_val = 10)
        # The loss is 1 - mean SSIM across the batch
        return 1.0 - tf.reduce_mean(ssim_value)
    return SSIMLoss

class CustomSSIMClass(tf.keras.losses.Loss):
    def __init__(self, xrange = [-1,1], log_standardize = True, name = 'CustomSSIMClass'):
        super().__init__(name=name)
        self.xrange = xrange
        self.log_standardize = log_standardize
        self.max_val=max(xrange)
        self.min_val=min(xrange)
    
    def call(self, y_true, y_pred):
        print('true max,min', tf.reduce_max(y_true), tf.reduce_min(y_true))
        if self.log_standardize:
            # print(y_true.shape, y_true.dtype, dir(y_true))
            y_target = log_scale(y_true)
            print('log max,min', tf.reduce_max(y_target), tf.reduce_min(y_target))
        else: 
            y_target = y_true
            # y_target = asymmetric_scaling(y_target, pos_scale = self.max_value, neg_scale = self.min_value)
        # max_val = max(max(y_true), max(y_pred))
        # Calculate SSIM for each image in the batch
        ssim_value = tf.image.ssim(y_target, y_pred, max_val=1)
        # The loss is 1 - mean SSIM across the batch
        return 1.0 - tf.reduce_mean(ssim_value)
        

def PSNRLoss(y_true, y_pred, max_val=3.0):
    psnr = tf.image.psnr(y_true, max_val*y_pred, max_val=max_val)
    return -tf.reduce_mean(psnr)
