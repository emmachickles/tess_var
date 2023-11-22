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
data_dir = '/scratch/echickle/cvae_data/'
sector_list = list(range(1,27))
# data_dir = '/scratch/data/tess/lcur/ffi/cvae_data/'
# sector_list = [56]

# Directory to save all outputs to
mydir = '/scratch/echickle/cvae_2min/'
os.makedirs(mydir, exist_ok=True)

# Hyperparameters
latent_dim = 4  # Size of the latent space
num_conv_layers = 5  # Number of convolutional layers in the encoder and decoder
num_filters = 32  # Number of filters in each convolutional layer
kernel_size = 3  # Kernel size for the convolutional layers
batch_size = 32
epochs=30
fully_connected=True

# Determine the desired length based on the model architecture
array_length = 128
# dense_length = array_length // (4 ** num_conv_layers) # >> length before Dense(latent_dim)
# desired_length = dense_length * (4 ** num_conv_layers)
dense_length = array_length // (2 ** num_conv_layers)
desired_length = dense_length * (2 ** num_conv_layers)
if desired_length != array_length:
    print('!! autoencoder not able to reconstruct to correct dimensions')
    pdb.set_trace()

# Load data
# flux_data = np.random.randn(60000, 128)
# ticid = np.arange(60000)
flux_data = np.load('/scratch/echickle/tess_binned_s0001-s0026_phase_curves.npy')
phases = flux_data[0][0]
light_curves = flux_data[:,:,1]
ticid = np.load('/scratch/echickle/tess_binned_s0001-s0026_ticid.npy')
desired_length=128

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('/scratch/echickle/hlsp_tess-svc_tess_lcf_all-s0001-s0026_tess_v1.0_cat.csv')
inds = np.nonzero(data['power'] > 0.1)
inds = np.nonzero( (data['power']>0.1) * data['period_var']<0.05) )
light_curves = light_curves[inds]
ticid = ticid[inds]

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

# X_train, X_test, ticid_train, ticid_test = train_test_split(light_curves, ticid, test_size=0.2, random_state=42)
# X_train, X_val, ticid_train, ticid_val = train_test_split(X_train, ticid_train, test_size=0.2, random_state=42)
# X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
# ticid_train, ticid_test, ticid_val = np.array(ticid_train), np.array(ticid_test), np.array(ticid_val)

X_train = X_train.reshape(-1, desired_length, 1)
X_val = X_val.reshape(-1, desired_length, 1)
X_test = X_test.reshape(-1, desired_length, 1)

# Define the encoder
inputs = Input(shape=(desired_length, 1))
x = inputs

if fully_connected:
    x = Flatten()(x)
    for _ in range(4):
        x = Dense(x.shape[1]/2)(x)
else:
    for _ in range(num_conv_layers):
        x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='elu', padding='same', strides=2,
                   kernel_regularizer=regularizers.l2(0.01))(x)
       # x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

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
if fully_connected:
    x = Dense(latent_inputs.shape[1]*2)(latent_inputs)
    for _ in range(4):
        x = Dense(x.shape[1]*2)(x)
    outputs=Reshape((x.shape[1], 1))(x)
else:
    x = Dense(dense_length * num_filters)(latent_inputs)
    x = Reshape((dense_length, num_filters))(x)
    for _ in range(num_conv_layers):
        x = Conv1DTranspose(filters=num_filters, kernel_size=kernel_size, activation='elu', padding='same', strides=2,  kernel_regularizer=regularizers.l2(0.01))(x)
        # x = UpSampling1D(size=2)(x)
    outputs = Conv1DTranspose(filters=1, kernel_size=kernel_size, activation='linear', padding='same')(x)

# Connect encoder and decoder
encoder = Model(inputs, [z_mean, z_log_var, z])
decoder = Model(latent_inputs, outputs)

encoder.summary()
decoder.summary()

# Connect the encoder and decoder to create the full CVAE model
outputs = decoder(encoder(inputs)[2])
cvae = Model(inputs, outputs)
cvae.summary()

pdb.set_trace()

# Define the loss function 
# def vae_loss(inputs, outputs):
#     reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
#     kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#     return reconstruction_loss + kl_loss

def vae_loss(inputs, outputs):
    reconstruction_loss = tf.reduce_sum(
        tf.square(inputs - outputs), axis=[1, 2]
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
latent_vectors = encoder.predict(X_test)[2]

plot_tsne(latent_vectors, mydir)

# Obtain the reconstructed data from the CVAE model
reconstructed_data = cvae.predict(X_test)

# Plot the original vs reconstructed light curves
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

kmeans_labels=cluster(latent_vectors, cluster_method='kmeans', n_clusters=15)
plot_cluster(latent_vectors, kmeans_labels, mydir)
plot_anomaly(latent_vectors, mydir)

data_test = data.iloc[idx_test]
label = 'Solution'
cluster_labels = np.unique(data_test[label])
label_values = np.array([np.nonzero(cluster_labels == sol)[0][0] for sol in data_test['Solution']])
plot_tsne_cmap(latent_vectors, label_values, label, mydir, cluster_labels=cluster_labels)

label = 'GAIAmag'
label_values = data_test[label].to_numpy()
inds = np.nonzero(~np.isnan(label_values))
plot_tsne_cmap(latent_vectors[inds], label_values[inds], label, mydir)

label = 'Teff'
label_values = data_test[label].to_numpy()
inds = np.nonzero(~np.isnan(label_values))
plot_tsne_cmap(latent_vectors[inds], label_values[inds], label, mydir)

plot_tsne_inset(latent_vectors, light_curves, kmeans_labels, mydir, dim=1)

movie_cluster(latent_vectors, kmeans_labels, mydir)    
# movie_light_curves(light_curves, mydir)

# plot_latent_images(decoder, 16, latent_dim, latent_vectors, mydir)

# Plot intermediate outputs
# visualize_layer_outputs(encoder, X_test, 4, mydir)


pdb.set_trace()

reconstruction_loss = np.mean(np.square(X_test - reconstructed_data), axis=1).reshape(-1)
inds = np.nonzero(reconstruction_loss < 0.1)[0]
X_test = X_test[inds]
reconstructed_data = reconstructed_data[inds]
latent_vectors = latent_vectors[inds]
light_curves = light_curves[inds]
data_test = data_test.iloc[inds]

plot_tsne(latent_vectors, mydir)

# Plot the original vs reconstructed light curves
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

kmeans_labels=cluster(latent_vectors, cluster_method='kmeans', n_clusters=15)
plot_cluster(latent_vectors, kmeans_labels, mydir)
plot_anomaly(latent_vectors, mydir)

label = 'Solution'
cluster_labels = np.unique(data_test[label])
label_values = np.array([np.nonzero(cluster_labels == sol)[0][0] for sol in data_test['Solution']])
plot_tsne_cmap(latent_vectors, label_values, label, mydir, cluster_labels=cluster_labels)

label = 'GAIAmag'
label_values = data_test[label].to_numpy()
inds = np.nonzero(~np.isnan(label_values))
plot_tsne_cmap(latent_vectors[inds], label_values[inds], label, mydir)

label = 'Teff'
label_values = data_test[label].to_numpy()
inds = np.nonzero(~np.isnan(label_values))
plot_tsne_cmap(latent_vectors[inds], label_values[inds], label, mydir)

plot_tsne_inset(latent_vectors, light_curves, kmeans_labels, mydir, dim=1)

movie_cluster(latent_vectors, kmeans_labels, mydir)    
