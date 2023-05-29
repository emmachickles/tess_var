import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Lambda, Reshape, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pdb



# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory containing the light curve data
data_dir = '/scratch/data/tess/lcur/spoc/clip/sector-01/'

# Hyperparameters
latent_dim = 10  # Size of the latent space
num_conv_layers = 2  # Number of convolutional layers in the encoder and decoder
num_filters = 32  # Number of filters in each convolutional layer
kernel_size = 3  # Kernel size for the convolutional layers

# Preprocess data
file_list = os.listdir(data_dir)
num_files = len(file_list)
time_array = np.load(os.path.join(data_dir, file_list[0]))[0]
max_time_length = time_array.shape[0]

# Load and preprocess light curve data
flux_data = []
for file_name in file_list:
    light_curve = np.load(os.path.join(data_dir, file_name))[1]

    # Interpolate nan values
    inds = np.nonzero(~np.isnan(light_curve))
    light_curve = np.interp(time_array, time_array[inds], light_curve[inds])

    # Reshape
    light_curve = np.expand_dims(light_curve, 0)

    # Standardize flux values
    scaler = StandardScaler()
    light_curve = scaler.fit_transform(light_curve)

    # Reshape
    light_curve = np.squeeze(light_curve, 0)

    flux_data.append(light_curve)

# Convert data to numpy array
flux_data = np.array(flux_data)
np.save('/scratch/echickle/sector_01_lc.npy', flux_data)
flux_data = np.load('/scratch/echickle/sector_01_lc.npy')
pdb.set_trace()

# Train-validation-test split
X_train, X_test = train_test_split(flux_data, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

# Define the encoder
inputs = Input(shape=(max_time_length, 1))
x = inputs
for _ in range(num_conv_layers):
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
x = Flatten()(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Define the decoder
latent_inputs = Input(shape=(latent_dim,))
x = Dense(max_time_length // (2 ** num_conv_layers) * num_filters)(latent_inputs)
x = Reshape((max_time_length // (2 ** num_conv_layers), num_filters))(x)
for _ in range(num_conv_layers):
    x = Conv1DTranspose(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
outputs = Conv1DTranspose(filters=1, kernel_size=kernel_size, activation='linear', padding='same')(x)

# Define the CVAE model
cvae = Model(inputs, outputs)

# Define the loss function
def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = mse(inputs, outputs)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconstruction_loss + kl_loss

# Compile the model
cvae.compile(optimizer='adam', loss=vae_loss)

# Train the CVAE
cvae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=50, batch_size=128)

# Evaluate the model on the test set
loss = cvae.evaluate(X_test, X_test)
print('Test Loss:', loss)
