import numpy as np
import tarfile
import h5py
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
from qkeras import QConv2D, QBatchNormalization
from qkeras.utils import _add_supported_quantized_objects
from qkeras import QDense, QActivation, QDenseBatchnorm, quantized_relu, quantized_bits
import argparse
import math
import os


def apply_power_of_2_scaling(X):
    result = [7, 5, 5, 7, 5, 5, 5, 4, 4, 5, 4, 3, 4, 3, 3, 3, 3, 2, 6, 5, 5, 5, 4, 5, 4, 3, 4, 3, 2, 3, 4, 4, 4, 2, 3, 3, 1, 1, 2, -2, -1, -1, 6, 5]

    # Apply the scaling using 2 raised to the power of the result
    X_scaled = X / (2.0 ** np.array(result))
    return X_scaled


def load_l1AD_model():

    # Define the custom objects dictionary for QKeras layers
    custom_objects = {
        'QDense': QDense,
        'QActivation': QActivation,
        'QDenseBatchnorm': QDenseBatchnorm,
        'quantized_relu': quantized_relu,
        'quantized_bits': quantized_bits
    }

    # Load the model with custom objects
    loaded_model = tf.keras.models.load_model(
        '/eos/home-m/mmcohen/ntuples/2A_AE_model_FDL_GAN_ALT_23e.h5',
        custom_objects=custom_objects,
        compile=False  # If they only need inference
    )

    return loaded_model

def main():

    parser = argparse.ArgumentParser(description='L1 AD inference')
    parser.add_argument(
        'ntuple_file',
        help='Path to the ntuple h5 file.'
    )
    
    args = parser.parse_args()

    # load the data
    with h5py.File(args.ntuple_file, 'r') as hf:
        L1_data = hf['L1_data'][:]

    
    # bitshift the data
    L1_data = apply_power_of_2_scaling(L1_data)

    # load the model
    model = load_l1AD_model()

    # predict the AD scores
    predictions = model.predict(L1_data, verbose=0)
    AD_scores = np.mean(np.square(predictions), axis=1)

    # Save the AD scores back to the h5 file
    with h5py.File(args.ntuple_file, 'a') as hf:
        if 'L1AD_scores' in hf:
            del hf['L1AD_scores']  # Remove existing dataset if it exists
        hf.create_dataset('L1AD_scores', data=AD_scores)


if __name__ == '__main__':
    main()

