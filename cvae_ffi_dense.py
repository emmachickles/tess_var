import os
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Lambda, Reshape, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import regularizers
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() # >> avoid KerasTensor TypeError issue
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import matplotlib
from plot_functions import *

matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.markerfacecolor'] = 'k'
matplotlib.rcParams['lines.markeredgecolor'] = 'k'
matplotlib.rcParams['lines.markersize'] = 1
matplotlib.rcParams['lines.linestyle'] = 'None'

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory containg preprocessed light curve data
data_dir = '/scratch/data/tess/lcur/ffi/cvae_data/'
sector_list = [61]
cam_list = [1]
ccd_list = [1,2,3,4]

# Directory to save all outputs to
mydir = '/scratch/echickle/cvae/'
os.makedirs(mydir, exist_ok=True)

# Hyperparameters
N = 128
num_layers = 5  # Number of fully-connected layers in the encoder and decoder
latent_dim = N // (2 ** num_layers) # Size of latent space = 4
batch_size = 32
epochs = 100

# Load data
mean_phi = np.linspace(0, 1-1/N, N) + 0.5/N
light_curves = []
# stat = []
ticid = []
for sector in sector_list:
    for cam in cam_list:
        for ccd in ccd_list:
            light_curves.extend(np.load(data_dir+'s%04d'%sector+'-cam{}-ccd{}.npy'.format(cam, ccd)))
            ticid.extend(np.load(data_dir+'s%04d'%sector+'-cam{}-ccd{}-ticid.npy'.format(cam, ccd)))
            # stat.extend(np.load(mydir+'s%04d'%sector+'-cam{}-ccd{}.npy'.format(cam, ccd)))  
light_curves = np.array(light_curves)  
ticid = np.array(ticid)

# 1d interpolate nans 
interpolated_lc = []
for lc in light_curves:
    if np.count_nonzero(np.isnan(lc)) > 0:
        inds = np.nonzero(~np.isnan(lc))
        new_lc = np.interp(mean_phi, mean_phi[inds], lc[inds])
        interpolated_lc.append(new_lc)
    else:
        interpolated_lc.append(lc)
light_curves = np.array(interpolated_lc)

# Standardize
mean = np.mean(light_curves, keepdims=True)
std = np.std(light_curves, keepdims=True)
light_curves = (light_curves - mean) / std 

# Train-validation-test split
X_train, X_test, idx_train, idx_test = train_test_split(light_curves, np.arange(len(light_curves)), test_size=0.2, random_state=42)
X_train, X_val, idx_train, idx_val = train_test_split(X_train, idx_train, test_size=0.2, random_state=42)
X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
idx_train, idx_val, idx_test = np.int64(np.array(idx_train)), np.int64(np.array(idx_val)), np.int64(np.array(idx_test))
ticid_train, ticid_val, ticid_test = ticid[idx_train], ticid[idx_val], ticid[idx_test]

# !! 
targ = 803489769
if targ in ticid_train:
    ind = np.nonzero(ticid_train == targ)[0]
    X_test = np.append(X_test, X_train[ind], axis=0)
    ticid_test = np.append(ticid_test, ticid_train[ind])
    X_train = np.delete(X_train, ind, axis=0)
    ticid_train = np.delete(ticid_train, ind)
if targ in ticid_val:
    ind = np.nonzero(ticid_val == targ)[0]
    X_test = np.append(X_test, X_val[ind], axis=0)
    ticid_test = np.append(ticid_test, ticid_val[ind])
    X_val = np.delete(X_val, ind, axis=0)
    ticid_val = np.delete(ticid_val, ind)


# Define the encoder
inputs = Input(shape=(N))
x = inputs

for _ in range(num_layers):
    x = Dense(x.shape[1]/2, activation='elu')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    latent_dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Define the decoder
latent_inputs = Input(shape=(latent_dim,))
x = Dense(latent_inputs.shape[1]*2)(latent_inputs)
for i in range(num_layers - 1):
    if i < num_layers - 2:
        x = Dense(x.shape[1]*2, activation='relu')(x)
    else:
        x = Dense(x.shape[1]*2)(x)
outputs=x

# Connect encoder and decoder
encoder = Model(inputs, [z_mean, z_log_var, z])
decoder = Model(latent_inputs, outputs)

encoder.summary()
decoder.summary()

# Connect the encoder and decoder to create the full CVAE model
outputs = decoder(encoder(inputs)[2])
cvae = Model(inputs, outputs)
cvae.summary()

def vae_loss(inputs, outputs):
    reconstruction_loss = tf.reduce_sum(
        tf.square(inputs - outputs), axis=[1]
    )  # Sum over the dimensions of the input data
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )  # Sum over the latent dimensions
    loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return loss

# Compile the model
cvae.compile(optimizer='adam', loss=vae_loss)

# Train the CVAE
history = cvae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs,
                   batch_size=batch_size) 

# Evaluate the model on the test set
loss = cvae.evaluate(X_test, X_test)
print('Test Loss:', loss)

plot_loss(history, mydir)

# Encode the test set samples
latent_vectors = encoder.predict(X_test)
z_mean = latent_vectors[0]
z_log_var = latent_vectors[1]

plot_tsne(z_mean, mydir)

# Obtain the reconstructed data from the CVAE model
reconstructed_data = cvae.predict(X_test)

# Plot the original vs reconstructed light curves
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

kmeans_labels=cluster(z_mean, cluster_method='kmeans', n_clusters=15)
plot_cluster(z_mean, kmeans_labels, mydir)
plot_anomaly(z_mean, X_test, ticid_test, mydir)
          
plot_tsne_inset(z_mean, X_test, kmeans_labels, mydir, dim=1)

indx = np.nonzero(ticid_test == 803489769)[0][0]
plot_tsne_kl(indx, z_mean, z_log_var, X_test, ticid_test, mydir, dim=1)
