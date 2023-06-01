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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.markerfacecolor'] = 'k'
matplotlib.rcParams['lines.markeredgecolor'] = 'k'
matplotlib.rcParams['lines.markersize'] = 1
matplotlib.rcParams['lines.linestyle'] = 'None'

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory containing the light curve data
data_dir = '/scratch/data/tess/lcur/spoc/clip/sector-01/'
mydir = '/scratch/echickle/cvae/'
os.makedirs(mydir, exist_ok=True)

# Hyperparameters
latent_dim = 10  # Size of the latent space
num_conv_layers = 5  # Number of convolutional layers in the encoder and decoder
num_filters = 32  # Number of filters in each convolutional layer
kernel_size = 3  # Kernel size for the convolutional layers
batch_size = 32

# Preprocess data
file_list = os.listdir(data_dir)
num_files = len(file_list)
time_array = np.load(os.path.join(data_dir, file_list[0]))[0]
num_inds = np.nonzero(~np.isnan(time_array))
time_array = time_array[num_inds]
max_time_length = time_array.shape[0]

# Load and preprocess light curve data
flux_data = []
for file_name in file_list:
    light_curve = np.load(os.path.join(data_dir, file_name))[1]
    light_curve = light_curve[num_inds]

    # Interpolate nan values
    inds = np.nonzero(~np.isnan(light_curve))
    light_curve = np.interp(time_array, time_array[inds], light_curve[inds])

    # Reshape
    light_curve = light_curve.reshape(-1, 1)

    # # Standardize flux values
    # scaler = StandardScaler()
    # Normalize flux values to standard range of [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    light_curve = scaler.fit_transform(light_curve)

    # Reshape
    light_curve = light_curve.flatten()

    flux_data.append(light_curve)

# Convert data to numpy array
flux_data = np.array(flux_data)
np.save('/scratch/echickle/sector_01_lc.npy', flux_data)
# flux_data = np.load('/scratch/echickle/sector_01_lc.npy')

# Determine the desired length based on the model architecture
dense_length = max_time_length // (4 ** num_conv_layers) + 1 # >> length before Dense(latent_dim)
desired_length = dense_length * (4 ** num_conv_layers)

# Pad the flux_data with zeros
padded_flux_data = []
for data in flux_data:
    if len(data) < desired_length:
        padding_length = desired_length - len(data)
        padded_data = np.pad(data, (0, padding_length), 'constant')
    else:
        padded_data = data[:desired_length]
    padded_flux_data.append(padded_data)

# Convert the padded_flux_data back to a numpy array
padded_flux_data = np.array(padded_flux_data)

# Train-validation-test split
X_train, X_test = train_test_split(padded_flux_data, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
X_train = X_train.reshape(-1, desired_length, 1)
X_val = X_val.reshape(-1, desired_length, 1)
X_test = X_test.reshape(-1, desired_length, 1)

# Define the encoder
inputs = Input(shape=(desired_length, 1))
x = inputs
for _ in range(num_conv_layers):
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='elu', padding='same', strides=2,
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(pool_size=2)(x)
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
    x = UpSampling1D(size=2)(x)
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
history = cvae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=10, batch_size=batch_size) 

# Extract loss history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plot loss vs epoch
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(mydir + 'loss.png')

# Evaluate the model on the test set
loss = cvae.evaluate(X_test, X_test)
print('Test Loss:', loss)

# Encode the test set samples
latent_vectors = encoder.predict(X_test)[2]

def plot_tsne(latent_vectors, mydir):
    """Create a scatter plot of the t-SNE latent space"""

    # Perform t-SNE dimensionality reduction on the latent vectors
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Extract the t-SNE coordinates
    tsne_x = latent_tsne[:, 0]
    tsne_y = latent_tsne[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_x, tsne_y, c='b', alpha=0.5)
    plt.title('t-SNE Latent Space Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(mydir+'tsne.png')

plot_tsne(latent_vectors, mydir)

def plot_latent_images(decoder, n, latent_dim, latent_vectors, mydir):
    """Plots n x n decoded images sampled from the latent space and saves the figure to mydir."""
    import tensorflow_probability as tfp

    norm = tfp.distributions.Normal(0, 1)
    grid_points = norm.quantile(np.linspace(0.05, 0.95, n * n))
    grid_points = tf.reshape(grid_points, (n * n, 1))
    z = tf.repeat(grid_points, repeats=latent_dim, axis=1)  # Repeat grid_points for each latent dimension
    x_decoded = decoder.predict(z, steps=1)  # Generate decoded images

    nrow = int(np.sqrt(n))
    ncol = int(np.sqrt(n))

    fig, ax = plt.subplots(nrow, ncol, figsize=(15,8)) # Plot light curves
    for i in range(nrow):
        for j in range(ncol):
            ax[i, j].plot(x_decoded[nrow*i + j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    fig.savefig(os.path.join(mydir, 'latent_images.png'))
    plt.close()

    # Perform t-SNE dimensionality reduction on the latent vectors and grid_points
    tsne = TSNE(n_components=2, random_state=42)
    all_vectors = np.append(latent_vectors, z.eval(session=tf.compat.v1.Session()), 0)
    latent_tsne = tsne.fit_transform(all_vectors)

    # Extract the t-SNE coordinates
    tsne_x = latent_tsne[:, 0]
    tsne_y = latent_tsne[:, 1]

    fig1, ax1 = plt.subplots(figsize=(8,6)) # Plot tsne
    ax1.set_title('t-SNE Latent Space Visualization')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.plot(tsne_x[:-z.shape[0]], tsne_y[:-z.shape[0]], 'k.', ms=1, alpha=0.5)
    for i in range(z.shape[0]):
        x, y = tsne_x[z.shape[0]+i], tsne_y[z.shape[0]+i]
        with plt.rc_context({'lines.markerfacecolor': 'b'}):
            ax1.plot(x, y, 'o', ms=10)
        row = i // nrow
        col = i % ncol
        ax1.annotate('{},{}'.format(row,col), (x, y+1.),
                     ha='center', c='k')
    fig1.savefig(os.path.join(mydir, 'latent_tsne.png'))
    plt.close()

plot_latent_images(decoder, 16, latent_dim, latent_vectors, mydir)

def plot_original_vs_reconstructed(original_data, reconstructed_data, num_examples, mydir):
    plt.figure(figsize=(20, 8))

    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)

    for i in range(num_examples):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[i])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[i])
        plt.title('Reconstructed\nLoss: '+str(np.round(reconstruction_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'original_vs_reconstructed.png'))
    plt.close()

def plot_best_worst_reconstructed(original_data, reconstructed_data, num_examples, mydir):
    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)
    worst_indices = np.argsort(reconstruction_loss)[-num_examples:][::-1]
    worst_loss = np.sort(reconstruction_loss)[-num_examples:][::-1]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[idx])
        plt.title('Reconstructed\nLoss: '+str(np.round(worst_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'worst_reconstructed.png'))
    plt.close()

    best_indices = np.argsort(reconstruction_loss)[:num_examples]
    best_loss = np.sort(reconstruction_loss)[:num_examples]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[idx])
        plt.title('Reconstructed\nLoss: '+str(np.round(best_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'best_reconstructed.png'))
    plt.close()

def plot_reconstruction_distribution(original_data, reconstructed_data, mydir):
    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)
    plt.figure()
    _ = plt.hist(reconstruction_loss, bins=100)
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Training Instances')
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'loss_dist.png'))

# Obtain the reconstructed data from the CVAE model
reconstructed_data = cvae.predict(X_test)

# Plot the original vs reconstructed light curves
plot_original_vs_reconstructed(X_test[:10], reconstructed_data[:10], 10, mydir)

# Plot worst reconstructions
plot_best_worst_reconstructed(X_test, reconstructed_data, 10, mydir)

# Plot reconstruction loss distribution
plot_reconstruction_distribution(X_test, reconstructed_data, mydir)

def visualize_layer_outputs(model, data, num_examples, mydir):
    layer_names = [layer.name for layer in model.layers[1:]]  # Get the names of all layers in the model
    layer_outputs = [layer.output for layer in model.layers[1:]]  # Get the output tensors of all layers

    # Create a new model that maps the input data to the output tensors of each layer
    intermediate_model = Model(inputs=model.input, outputs=layer_outputs)

    # Get the intermediate layer outputs for the given data
    intermediate_outputs = intermediate_model.predict(data)

    # Plot and save the output from each layer
    for layer_name, intermediate_output in zip(layer_names, intermediate_outputs):
        if len(intermediate_output.shape) == 3:  # Handle 1D convolutional layer outputs
            num_filters = intermediate_output.shape[-1]
            fig, axes = plt.subplots(num_filters, num_examples, figsize=(16, 16))
            fig.suptitle(layer_name)

            for i in range(num_filters):
                for j in range(num_examples):
                    axes[i,j].plot(intermediate_output[j, :, i])
                    axes[i,j].set_xticks([])
                    # axes[i].set_yticks([])

            plt.savefig(os.path.join(mydir, f'{layer_name}.png'))
            plt.close()

        elif len(intermediate_output.shape) == 2:  # Handle dense layer outputs
            fig, ax = plt.subplots(ncols=num_examples, figsize=(16, 5))
            fig.suptitle(layer_name)

            for i in range(num_examples):
                ax[i].plot(intermediate_output[i])
                ax[i].set_xticks([])
                # ax.set_yticks([])

            plt.savefig(os.path.join(mydir, f'{layer_name}.png'))
            plt.close()

visualize_layer_outputs(encoder, X_test, 4, mydir)

pdb.set_trace()

