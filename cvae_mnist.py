import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Lambda, Reshape, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import regularizers
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() # >> avoid KerasTensor TypeError issue
from scipy.ndimage import median_filter
from scipy.interpolate import interp2d
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

# Directory to save all outputs to
# mydir = '/Users/emma/Desktop/230801/'
mydir = '/scratch/echickle/cvae_mnist/'
os.makedirs(mydir, exist_ok=True)

# Hyperparameters
latent_dim = 4  # Size of the latent space
num_conv_layers = 5  # Number of convolutional layers in the encoder and decoder
num_filters = 32  # Number of filters in each convolutional layer
kernel_size = 3  # Kernel size for the convolutional layers
batch_size = 32
desired_length = 256 # original MNIST shape =28*28=784
dense_length = desired_length // (2 ** num_conv_layers)
beta = 1
epochs = 20
imbalance = True

# Load MNIST dataset
# Training images: (60000, 28, 28)
# Training labels: (60000,)
# Testing images: (10000, 28, 28)
# Testing labels: (10000,)
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

if imbalance:
    inds = np.nonzero(Y_train == 1)[0]
    X_train = np.delete(X_train, inds[500:], axis=0)
    Y_train = np.delete(Y_train, inds[500:])

    inds = np.nonzero(Y_train == 2)[0]
    X_train = np.delete(X_train, inds[500:], axis=0)
    Y_train = np.delete(Y_train, inds[500:])
    
fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(16,4))
for i in range(10):
    ax[0][i].imshow( X_train[ np.nonzero(Y_train==i)[0][0] ] )

# 2D Interpolate    
old_x = np.arange(28)
old_y = np.arange(28)
new_x = np.linspace(0, 28-1, 16)
new_y = np.linspace(0, 28-1, 16)
X_train_interp = []
for i in range(len(X_train)):
    X_train_interp.append( interp2d(old_x, old_y, X_train[i])(new_x, new_y) )
X_train = np.array(X_train_interp).astype('float32')
X_test_interp = []
for i in range(len(X_test)):
    X_test_interp.append( interp2d(old_x, old_y, X_test[i])(new_x, new_y) )
X_test = np.array(X_test_interp).astype('float32')

for i in range(10):
    ax[1][i].imshow( X_train[ np.nonzero(Y_train==i)[0][0] ] )
plt.tight_layout()
fig.savefig(mydir+'dataset.png', dpi=300)

# Flatten images
X_train = X_train.reshape((-1, 256))
X_test = X_test.reshape((-1, 256))

# Standardize
mean = np.mean(X_train, axis=1, keepdims=True)
std = np.std(X_train, axis=1, keepdims=True)
X_train = (X_train - mean) / std 
mean = np.mean(X_test, axis=1, keepdims=True)
std = np.std(X_test, axis=1, keepdims=True)
X_test = (X_test - mean) / std 

# Add additional axis
val_split = len(X_test) // 2
X_val = X_test[:val_split]
Y_val = Y_test[:val_split]
X_test = X_test[val_split:]
Y_test = Y_test[val_split:]
X_train = X_train.reshape(-1, desired_length, 1)
X_val = X_val.reshape(-1, desired_length, 1)
X_test = X_test.reshape(-1, desired_length, 1)


# Define the encoder
inputs = Input(shape=(desired_length, 1))
x = inputs
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

def vae_loss(inputs, outputs):
    reconstruction_loss = tf.reduce_sum(
        tf.square(inputs - outputs), axis=[1, 2]
    )  # Sum over the dimensions of the input data
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )  # Sum over the latent dimensions
    loss = tf.reduce_mean(reconstruction_loss + beta*kl_loss)
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
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir, dim=2, cycles=1)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir, dim=2)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

plot_latent_images(decoder, 25, latent_dim, latent_vectors, mydir, dim=2)

kmeans_labels=cluster(latent_vectors, cluster_method='kmeans', n_clusters=10)
plot_cluster(latent_vectors, kmeans_labels, mydir)
plot_anomaly(latent_vectors, mydir)

plot_tsne_cmap(latent_vectors, Y_test, 'Y_test', mydir)
# movie_cluster(latent_vectors, kmeans_labels, mydir)    
plot_tsne_inset(latent_vectors, X_test, kmeans_labels, mydir, dim=2)


# # Plot intermediate outputs
# visualize_layer_outputs(encoder, X_test, 4, mydir)

pdb.set_trace()

# Make a cut based on loss
reconstruction_loss = np.mean(np.square(X_test - reconstructed_data), axis=1).reshape(-1)
inds = np.nonzero(reconstruction_loss < 0.4)[0]
X_test, Y_test = X_test[inds], Y_test[inds]
reconstructed_data = reconstructed_data[inds]
latent_vectors = latent_vectors[inds]


# Plot the original vs reconstructed light curves
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir, dim=2, cycles=1)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir, dim=2)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

plot_latent_images(decoder, 25, latent_dim, latent_vectors, mydir, dim=2)

kmeans_labels=cluster(latent_vectors, cluster_method='kmeans', n_clusters=10)
plot_cluster(latent_vectors, kmeans_labels, mydir)
plot_anomaly(latent_vectors, mydir)

plot_tsne_cmap(latent_vectors, Y_test, 'Y_test', mydir)
# movie_cluster(latent_vectors, kmeans_labels, mydir)    
plot_tsne_inset(latent_vectors, X_test, kmeans_labels, mydir, dim=2)

