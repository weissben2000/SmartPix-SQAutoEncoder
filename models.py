import keras
import math
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *
import os
# from dataset_utils import quantize_manual

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from SoftQuantizeLayer import SoftQuantizeLayer

def hard_quantize(x, levels, thresholds):
        x_reshaped = tf.expand_dims(x, axis=-1) 
        is_grt_th = x_reshaped > thresholds 
        indices = tf.reduce_sum(tf.cast(is_grt_th, dtype=tf.int32), axis=-1) 
        return tf.gather(levels, indices)

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, kernel_size=3, n_filters=5):
    var = QSeparableConv2D(
        n_filters, kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters, 1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def CreateModel(shape, n_filters, pool_size, conv_kernel_size, mean_filter=False, thresh = False):
    x_base = x_in = Input(shape)
    if mean_filter:
        stack = AveragePooling2D(
            pool_size=(mean_filter,mean_filter), 
            strides = (1, 1), 
            padding = "same",
             #data_format = "channels_first"
        )(x_base)
        if thresh:
            apply_thresh = lambda x: tf.where(x < thresh, tf.zeros_like(x), x)
            stack = Lambda(function=apply_thresh)(stack)
        stack = conv_network(stack, kernel_size=conv_kernel_size)
    else:
        stack = conv_network(x_base, kernel_size=conv_kernel_size)
    
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQMaxconv2DModel(shape, n_filters = 5, pool_size = 3, conv_kernel_size = 3, mean_filter=False, thresh = False, SQAutoEncode = False):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        # initial_thresholds = (np.arange(3)+1,
        initial_thresholds = [10,240, 400],

        # initial_levels = [0.0, 0.33333, 0.66667, 1.0],
        initial_levels = np.arange(4),
        threshold_offset = 0,
        # initial_range=[-1.0, 1.0],
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    if mean_filter:
        stack = AveragePooling2D(
            pool_size=(mean_filter,mean_filter), 
            strides = (1, 1), 
            padding = "same",
             #data_format = "channels_first"
        )(x_base)
        if thresh:
            apply_thresh = lambda x: tf.where(x < thresh, tf.zeros_like(x), x)
            stack = Lambda(function=apply_thresh)(stack)
        stack = conv_network(stack, kernel_size=conv_kernel_size)
    else:
        stack = conv_network(x_base, kernel_size=conv_kernel_size)
    
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    if SQAutoEncode:
        for layer in stack.layers:
            layer.trainable = False
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model
# import models
# importlib.reload(models)
# from models import CreateAutoEncoder
# from SoftQuantizeLayer import SoftQuantizeLayer
# from AnnealingScheduler import AnnealingScheduler

def encoder_network(shape, var, kernel_size=3, n_filters=5, pool_size=2):
    var = QSeparableConv2D(
        n_filters, kernel_size, padding = 'same',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=pool_size, 
        padding="same", 
        data_format='channels_last',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,kernel_size, padding = 'same',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)  
    var = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=pool_size, 
        padding="same", 
        data_format='channels_last',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)  
    return var

def decoder_network(shape, var, kernel_size=3, n_filters=5, pool_size=2):
    var = Conv2DTranspose(
        n_filters, 1, padding = 'same', strides = pool_size,
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = Conv2DTranspose(
        n_filters, kernel_size,
        strides = pool_size,
        padding="same",
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)    
    var = Conv2D(shape[-1], (pool_size, pool_size), padding="same")(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    # if input_quant is not None:
    #     # quantize = lambda x: quantize_manual(x, charge_levels = input_quant, quant_values =[0,1,2,3])
    #     quantize = lambda x: hard_quantize(x, levels = [0.0,1.0,2.0,3.0], thresholds = input_quant)
    #     var = Lambda(function=quantize, dtype=tf.float32)(var)
    return var

def Qdecoder_network(shape, var, Out_bits = 2,  kernel_size=3, n_filters=5, pool_size=2, conv2d=False):
    var = Conv2DTranspose(
        n_filters, 1, padding = 'same', strides = pool_size,
    )(var)
    if conv2d: 
        var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = Conv2DTranspose(
        n_filters, kernel_size,
        strides = pool_size,
        padding="same",
    )(var)
    if conv2d: 
        var = QActivation("quantized_tanh(16, 0, 1)")(var)    
    var = Conv2D(shape[-1], (pool_size, pool_size), padding="same")(var)
    if conv2d:
        var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = Flatten()(var)
    var = Dense(math.prod(shape)*(2**Out_bits), #activation='softmax'
               )(var)
    var = Reshape((*shape, 2**Out_bits), name="SQAutoEncoder_output")(var)
    return var

def get_min_max(training_generator, validation_generator, logrange=True):
    if logrange:
        tmax,tmin = [training_generator.dataset_logmax, training_generator.dataset_logmin]
        vmax,vmin = [validation_generator.dataset_logmax, validation_generator.dataset_logmin]
    else:
        tmax,tmin = [training_generator.dataset_max, training_generator.dataset_min]
        vmax,vmin = [validation_generator.dataset_max, validation_generator.dataset_min]
    return np.float32(min(tmin, vmin)), np.float32(max(tmax, vmax))

def CreateSQAutoEncoder(shape, FEBits = 2, OutBits = 3, n_filters=5, pool_size=2, 
                        conv_kernel_size=3, decoder_filters = 5, mean_filter=False, 
                        thresh = False, xrange = [-1,1], log_standardize=True):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=FEBits,
        initial_thresholds = (np.arange(2**FEBits - 1)+1)*100,
        # initial_thresholds = [0,240, 400],

        # initial_levels = [0.0, 0.33333, 0.66667, 1.0],
        initial_levels = np.arange(2**FEBits),
        threshold_offset = 0,
        # initial_range=[-1.0, 1.0],
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = encoder_network(shape, x_base, n_filters=n_filters, pool_size=pool_size, kernel_size = conv_kernel_size)
    stack = Qdecoder_network(shape, stack, Out_bits = OutBits, n_filters=decoder_filters, pool_size=pool_size, kernel_size = conv_kernel_size)
    # print(type(stack))
    if xrange is not None:
        def scaled_function(data):
            return asymmetric_scaling(data, pos_scale=xrange[1], neg_scale=abs(xrange[0]))
        stack = Lambda(function = scaled_function)(stack)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQModel_Conv2DSlim(shape, n_filters, pool_size, levels=[0,1,2,3], initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQAutoencoder_Conv2D(shape, n_filters, pool_size, levels=[0,1,2,3], initial_thresholds=[10,100,200], decoder_filters = 64, OutBits = 2, conv_kernel_size=3):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = conv_network(x_base, n_filters = n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = Qdecoder_network(shape, stack, Out_bits = OutBits, n_filters=decoder_filters, pool_size=2, kernel_size = conv_kernel_size, conv2d=True)
    model = Model(inputs=x_in, outputs=stack)
    return model
    

def CreateSQAE_MaxReg_fromweights(shape, weights_path, FEBits = 2, OutBits = 3, n_filters=5, pool_size=2, 
                        conv_kernel_size=3, decoder_filters = 5):
    # weights_path = './SQ_DAE/model-e0e43247-SQDAE_Categorical_2bit_240eN_Target_0_30_856-checkpoints'
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=CreateSQAutoEncoder((16,16,2), FEBits = FEBits, OutBits = OutBits, n_filters=5, pool_size=2,
                            conv_kernel_size = 3, decoder_filters = 128, xrange = None)
    
    DAE.load_weights(latest_checkpoint)
    # model_summary = [[layer.name, layer.output.shape] for layer in DAE.layers]
    
    # for i, layer in enumerate(DAE.layers):
    #     print('index:', i, layer.name)
    
    
    layer_name = DAE.layers[9].name 
    
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape

    print('latent space layer: ', layer_name, lat_shape)
    dense = var_network(encoder.output, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=dense)
    return model

def CreateSQAE_SlimReg_fromweights(shape, weights_path, FEBits = 2, OutBits = 3, n_filters=5, pool_size=2, 
                        conv_kernel_size=3, decoder_filters = 5):
    # weights_path = './SQ_DAE/model-e0e43247-SQDAE_Categorical_2bit_240eN_Target_0_30_856-checkpoints'
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=CreateSQAutoEncoder((16,16,2), FEBits = FEBits, OutBits = OutBits, n_filters=5, pool_size=2,
                            conv_kernel_size = 3, decoder_filters = 128, xrange = None)
    
    DAE.load_weights(latest_checkpoint)
    
    layer_name = DAE.layers[9].name 
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape
    print('latent space layer: ', layer_name, lat_shape)
    
    dense = var_network(encoder.output, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=dense)
    return model

def CreateSQAE_MaxConv2DReg_fromweights(shape, weights_path, FEBits = 2, OutBits = 3, 
                                  n_filters=5, pool_size=3,  conv_kernel_size=3, 
                                  decoder_filters = 5):
    # weights_path = './SQ_DAE/model-e0e43247-SQDAE_Categorical_2bit_240eN_Target_0_30_856-checkpoints'
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=CreateSQAutoencoder_Conv2D((16,16,2), n_filters=n_filters, pool_size=pool_size,
                        decoder_filters = 128, initial_thresholds=[10,80,800], 
                                     OutBits = OutBits, conv_kernel_size = conv_kernel_size)
    
    DAE.load_weights(latest_checkpoint)
    model_summary = [[layer.name, layer.output.shape] for layer in DAE.layers]
    
    # for i, layer in enumerate(DAE.layers):
    #     print('index:', i, layer.name)
    
    layer_name = DAE.layers[7].name 
    
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape

    print('latent space layer: ', layer_name, lat_shape)
    dense = var_network(encoder.output, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=dense)
    return model

def CreateSQAE_SlimConv2DReg_fromweights(shape, weights_path, FEBits = 2, OutBits = 3, 
                                  n_filters=5, pool_size=3,  conv_kernel_size=3, 
                                  decoder_filters = 5):
    # weights_path = './SQ_DAE/model-e0e43247-SQDAE_Categorical_2bit_240eN_Target_0_30_856-checkpoints'
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=CreateSQAutoencoder_Conv2D((16,16,2), n_filters=n_filters, pool_size=pool_size,
                        decoder_filters = 128, initial_thresholds=[10,80,800], 
                                     OutBits = OutBits, conv_kernel_size = conv_kernel_size)

    SQ_layer = DAE.get_layer('soft_quantizer_output')
    SQ_weights = SQ_layer.get_weights()
    print(SQ_weights)
    
    DAE.load_weights(latest_checkpoint)
    # model_summary = [[layer.name, layer.output.shape] for layer in DAE.layers]
    
    # for i, layer in enumerate(DAE.layers):
    #     print('index:', i, layer.name)
    SQ_layer = DAE.get_layer('soft_quantizer_output')
    SQ_weights = SQ_layer.get_weights()
    print(SQ_weights)
    
    layer_name = DAE.layers[7].name 
    
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape

    print('latent space layer: ', layer_name, lat_shape)
    dense = var_network(encoder.output, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=dense)
    return model

    # Trasformer model

class PatchExtractor(layers.Layer):
    """Extract 2D patches from images."""
    def __init__(self, patch_size=(3,7)):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # images: (batch, H, W, C)
        patch_h, patch_w = self.patch_size
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, patch_h, patch_w, 1),
            rates=(1,1,1,1),
            padding='VALID')
        # Now `patches` has shape: 
        #   (batch, H//patch_h, W//patch_w, patch_h*patch_w*C)
        # Flatten the 2D grid of patches => (batch, num_patches, patch_dim)
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(
            patches,
            [batch_size, -1, patch_dims]
        )
        return patches

class PatchEncoder(layers.Layer):
    """Linear embedding + learnable positional encoding."""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection  = layers.Dense(embed_dim)
        self.pos_embed   = tf.Variable(
            initial_value=tf.zeros((1,num_patches,embed_dim)),
            trainable=True,
            name="pos_embedding"
        )

    def call(self, patch_batch):
        # Linear projection of patch tokens
        projected = self.projection(patch_batch)
        # Add learnable positional embeddings
        return projected + self.pos_embed

def transformer_encoder(inputs,
                        head_size,
                        num_heads,
                        ff_dim,
                        dropout=0.1):
    # LayerNorm + Multi-head attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads,
                                  key_dim=head_size,
                                  dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
  
    # Next LN + feed-forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation="linear")(x)
    x = layers.Dropout(dropout)(x)
  
    return x + res

def create_vit_model(input_shape=(16,16,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     threshold_offset=0.0,
                     head_dims=(256,128,64),
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=14):
    # Input
    inp = layers.Input(shape=input_shape, name="raw_input")
    
    # q_out = SoftQuantizeLayer(
    #     n_bits=2,
    #     initial_range=[-1.0, 1.0],
    #     trainable_levels=False,
    #     trainable_thresholds=True,
    #     initial_k=1.0,
    #     trainable_k=True,
    #     name="soft_quantizer_output"
    # )(inp)
    
    q_out = SoftQuantizeLayer(
          n_bits=n_bits,
         #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
          initial_levels=np.array([i for i in range(0,np.power(2,n_bits))], dtype=np.float32),
          threshold_offset=threshold_offset,
          initial_thresholds=initial_thresholds,
          trainable_levels=False,
          trainable_thresholds=True,
          initial_k=1.0,
          trainable_k=False,
          name="soft_quantizer_output"
    )(inp)
    
    # 1) Extract patches
    patches = PatchExtractor(patch_size=patch_size)(q_out)
    
    # Calculate how many patches we extracted:
    #   (H // patch_h) * (W // patch_w)
    # Must do it statically if possible:
    # e.g. 13//3=4, 21//7=3 => 12 patches total
    H, W, C = input_shape
    ph, pw  = patch_size
    num_patches = (H // ph) * (W // pw)
    
    # 2) Encode patches (linear projection + positional embedding)
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)
    
    # 3) Apply multiple Transformer encoder blocks
    x = encoded_patches
    for _ in range(num_layers):
        x = transformer_encoder(x,
                                head_size=embed_dim,
                                num_heads=num_heads,
                                ff_dim=ff_dim,
                                dropout=dropout
                               )
        
    # 4) Flatten and final Dense
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # Regression head. The default (64,) is the legacy single-bottleneck head:
    # Flatten(1280) -> Dense(64) -> Dense(output), i.e. ONE hidden layer holding
    # 19.6% of the model's parameters in a single matrix, shared by all 14
    # Gaussian parameters. head_dims=(256,128,64) is the Deep variant (option
    # O17): same trunk, deeper funnel, testing whether the shallow shared
    # bottleneck is what makes good basins rare (measured on frozen [11,26]:
    # only ~1 in 5 seeds reaches the deep NLL basin).
    for w in head_dims:
        x = layers.Dense(w, activation="relu")(x)
    # outputs = layers.Dense(output, activation="linear")(x)
    outputs = layers.Dense(final_outputs, activation='linear')(x)
    
    # Create model
    model = keras.Model(inputs=inp, outputs=outputs)
    return model

def Create_SQDAE_transformer_model(input_shape=(16,16,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     threshold_offset=0.0,
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=3,
                     Out_bits = 2, 
                     n_filters=128, 
                     conv_kernel_size = 3
                    ):
    # Input
    inp = layers.Input(shape=input_shape, name="raw_input")
    
    # q_out = SoftQuantizeLayer(
    #     n_bits=2,
    #     initial_range=[-1.0, 1.0],
    #     trainable_levels=False,
    #     trainable_thresholds=True,
    #     initial_k=1.0,
    #     trainable_k=True,
    #     name="soft_quantizer_output"
    # )(inp)
    
    q_out = SoftQuantizeLayer(
          n_bits=n_bits,
         #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
          initial_levels=np.array([i for i in range(0,np.power(2,n_bits))], dtype=np.float32),
          threshold_offset=threshold_offset,
          initial_thresholds=initial_thresholds,
          trainable_levels=False,
          trainable_thresholds=True,
          initial_k=1.0,
          trainable_k=False,
          name="soft_quantizer_output"
    )(inp)
    
    # 1) Extract patches
    patches = PatchExtractor(patch_size=patch_size)(q_out)
    
    # Calculate how many patches we extracted:
    #   (H // patch_h) * (W // patch_w)
    # Must do it statically if possible:
    # e.g. 13//3=4, 21//7=3 => 12 patches total
    H, W, C = input_shape
    ph, pw  = patch_size
    num_patches = (H // ph) * (W // pw)
    
    # 2) Encode patches (linear projection + positional embedding)
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)
    
    # 3) Apply multiple Transformer encoder blocks
    x = encoded_patches
    for _ in range(num_layers):
        x = transformer_encoder(x,
                                head_size=embed_dim,
                                num_heads=num_heads,
                                ff_dim=ff_dim,
                                dropout=dropout)
    
    # 4) Convert 20 x 64 (patches x embed dims) to 4x4x64
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Permute((2, 1))(x)          # Shape becomes (64, 20)
    x = layers.Dense(16)(x)                    # Shape becomes (64, 16)
    x = layers.Permute((2, 1))(x)              # Shape becomes (16, 64)
    x = layers.Reshape((4, 4, 64))(x)

    # 5) Attach SQAutoEncoder decoder

    x = Qdecoder_network(input_shape, x, Out_bits = Out_bits, n_filters=n_filters, pool_size=2, kernel_size = conv_kernel_size, conv2d=True)
    model = Model(inputs=inp, outputs=x)
    return model

def Create_2head_xformer_model(input_shape=(16,16,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     head_dims=(256,128,64),
                     threshold_offset=0.0,
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=14,
                     Out_bits = 2, 
                     n_filters=128, 
                     conv_kernel_size = 3
                    ):
    # Input
    inp = layers.Input(shape=input_shape, name="raw_input")
    
    q_out = SoftQuantizeLayer(
          n_bits=n_bits,
         #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
          initial_levels=np.array([i for i in range(0,np.power(2,n_bits))], dtype=np.float32),
          threshold_offset=threshold_offset,
          initial_thresholds=initial_thresholds,
          trainable_levels=False,
          trainable_thresholds=True,
          initial_k=1.0,
          trainable_k=False,
          name="soft_quantizer_output"
    )(inp)
    
    # 1) Extract patches
    patches = PatchExtractor(patch_size=patch_size)(q_out)
    
    # Calculate how many patches we extracted:
    #   (H // patch_h) * (W // patch_w)
    # Must do it statically if possible:
    # e.g. 13//3=4, 21//7=3 => 12 patches total
    H, W, C = input_shape
    ph, pw  = patch_size
    num_patches = (H // ph) * (W // pw)
    
    # 2) Encode patches (linear projection + positional embedding)
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)
    
    # 3) Apply multiple Transformer encoder blocks
    x = encoded_patches
    for _ in range(num_layers):
        x = transformer_encoder(x,
                                head_size=embed_dim,
                                num_heads=num_heads,
                                ff_dim=ff_dim,
                                dropout=dropout)
    
    # 4) Convert 20 x 64 (patches x embed dims) to 4x4x64
    x_int = layers.LayerNormalization(epsilon=1e-6)(x)
    x_int = layers.Permute((2, 1))(x_int)          # Shape becomes (64, 20)
    x_int = layers.Dense(16)(x_int)                    # Shape becomes (64, 16)
    x_int = layers.Permute((2, 1))(x_int)              # Shape becomes (16, 64)
    x_int = layers.Reshape((4, 4, 64))(x_int)

    # 5) Attach SQAutoEncoder decoder
    SQAutoEncoder = Qdecoder_network(input_shape, x_int, Out_bits = Out_bits, n_filters=n_filters, pool_size=2, kernel_size = conv_kernel_size, conv2d=True)
    
    # 6) attach regression network as a second head
    reg = layers.Flatten()(x)
    for w in head_dims:
        reg = layers.Dense(w, activation="relu")(reg)
    
    reg = layers.Dense(final_outputs, activation="linear", name="reg_output")(reg)
    model = Model(inputs=inp, outputs={'SQAutoEncoder_output':SQAutoEncoder, 'reg_output':reg})
    return model
    
def CreateSQAE_MaxTransformerReg_fromweights(weights_path,
                     input_shape=(16,16,2),
                     patch_size=(3,4),
                     head_dims=(256,128,64),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     threshold_offset=0.0,
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=3):
  # Input
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=Create_SQDAE_transformer_model(input_shape=(16,16,2),   
                             patch_size=(3,4),        
                             embed_dim=embed_dim,           
                             num_heads=num_heads,            
                             ff_dim=ff_dim,              
                             num_layers=num_layers,            
                             dropout=dropout,        
                             threshold_offset=threshold_offset,
                             n_bits=n_bits,
                             initial_thresholds=initial_thresholds,
                             final_outputs=14         
                            )
    DAE.load_weights(latest_checkpoint)
    # SQ_layer = DAE.get_layer('soft_quantizer_output')
    # SQ_weights = SQ_layer.get_weights()
    # print(SQ_weights)
    layer_name = DAE.layers[44].name
    # layer_name = DAE.get_layer('layer_normalization').name 
    print(layer_name)
    
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape
    print(lat_shape)
        
    x = layers.Flatten()(encoder.output)
    for w in head_dims:
        x = layers.Dense(w, activation="relu")(x)
    outputs = layers.Dense(final_outputs, activation="linear")(x)
    
    # Create model
    model = keras.Model(inputs=x_in, outputs=outputs)
    return model
    # x = layers.Dense(64, activation='relu')(x)
    
    # outputs = layers.Dense(final_outputs, activation='linear')(x)
    
    # # Create model
    # model = keras.Model(inputs=x_in, outputs=outputs)
    # return model

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten()(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten()(proj_y)

    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = QActivation("quantized_relu(bits=13, integer=5)")(proj_x)

    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = QActivation("quantized_relu(bits=13, integer=5)")(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return var

def CreateSQModel_MLPSlim(shape, initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = [0,1,2,3],
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = mlp_encoder_network(x_base)

    stack = var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQAutoEncoder_MLPSlim(shape, 
                                initial_thresholds=[10,100,200], 
                                Out_bits = 2, 
                                n_filters=128, 
                                conv_kernel_size = 3):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = [0,1,2,3],
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = mlp_encoder_network(x_base)
    # print(stack.shape)
    x = layers.Dense(80)(stack)
    x = layers.Reshape((4, 4, 5))(x)
    x = Qdecoder_network(shape, x, Out_bits = Out_bits, n_filters=n_filters, pool_size=2, kernel_size = conv_kernel_size, conv2d=True)
    model = Model(inputs=x_in, outputs=x)
    return model

def CreateSQAE_SlimMLPReg_fromweights(weights_path,
                     input_shape=(16,16,2),
                     threshold_offset=0.0,
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=3):
  # Input
    base_dir = weights_path
    checkpoint_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_checkpoint}")
    DAE=CreateSQAutoEncoder_MLPSlim(shape = (16,16,2), 
                          # n_filters=128,
                          # pool_size=3,
                          initial_thresholds = initial_thresholds)
    DAE.load_weights(latest_checkpoint)
    # SQ_layer = DAE.get_layer('soft_quantizer_output')
    # SQ_weights = SQ_layer.get_weights()
    # print(SQ_weights)
    layer_name = DAE.layers[12].name
    # layer_name = DAE.get_layer('layer_normalization').name 
    print(layer_name)
    
    encoder = Model(inputs=DAE.input, outputs=DAE.get_layer(layer_name).output)
    encoder.trainable = False

    x_in = encoder.input
    lat_shape = encoder.output.shape
    print('latent space layer: ', layer_name, lat_shape)
    
    dense = var_network(encoder.output, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=dense)
    return model
    
    # x_in = encoder.input
    # lat_shape = encoder.output.shape
    # print(lat_shape)
        
    # x = layers.Flatten()(encoder.output)
    # x = layers.Dense(64, activation='relu')(x)
    # outputs = layers.Dense(final_outputs, activation='linear')(x)
    
    # # Create model
    # model = keras.Model(inputs=x_in, outputs=outputs)
    # return model