import os
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.markerfacecolor'] = 'k'
matplotlib.rcParams['lines.markeredgecolor'] = 'k'
matplotlib.rcParams['lines.markersize'] = 1
matplotlib.rcParams['lines.linestyle'] = 'None'

mydir = '/scratch/echickle/cvae_data/'
os.makedirs(mydir, exist_ok=True)

sector_list = [1]
# sector_list = list(range(1,27))

# Hyperparameters
latent_dim = 10  # Size of the latent space
num_conv_layers = 5  # Number of convolutional layers in the encoder and decoder
num_filters = 32  # Number of filters in each convolutional layer
kernel_size = 3  # Kernel size for the convolutional layers
batch_size = 32

# Preprocessing hyperparameters
window_size = 15
sigma_threshold = 5

# Determine the maximum possible number of measurements in a light curve
length_array = []
for sector in sector_list:
    data_dir = '/scratch/data/tess/lcur/spoc/mask/sector-%02d/'%sector
    file_list = os.listdir(data_dir)
    time_array = np.load(data_dir+file_list[0])[0]
    length_array.append(np.count_nonzero(~np.isnan(time_array)))
max_time_length = max(length_array)

# Determine the desired length based on the model architecture
dense_length = max_time_length // (4 ** num_conv_layers) + 1 # >> length before Dense(latent_dim)
desired_length = dense_length * (4 ** num_conv_layers)

fig, ax = plt.subplots(3, figsize=(8,8))
for sector in sector_list:
    print(sector)

    # Directory containing the light curve data
    data_dir = '/scratch/data/tess/lcur/spoc/mask/sector-%02d/'%sector # Quality flags masked out

    file_list = os.listdir(data_dir)

    # Load and preprocess light curve data
    flux_data = []
    for file_name in [file_list[861]]:
        light_curve = np.load(data_dir+file_name)[1]
        ax[0].plot(light_curve)

        # Remove nan values
        processed_light_curve = np.zeros_like(light_curve)
        num_inds = np.where(~np.isnan(light_curve))[0] 
        light_curve = light_curve[num_inds]

        # Apply rolling median filter to the flux data
        rolling_median = median_filter(light_curve, size=window_size, mode='reflect')

        # Calculate the rolling standard deviation
        rolling_std = np.std(light_curve - rolling_median)

        # Define the upper and lower clip thresholds
        upper_threshold = rolling_median + sigma_threshold * rolling_std
        lower_threshold = rolling_median - sigma_threshold * rolling_std

        # Perform sigma clipping
        clipped_indices = np.where((light_curve < upper_threshold) & (light_curve > lower_threshold))[0]

        # Return the clipped flux array
        clipped_light_curve = np.zeros_like(light_curve)
        light_curve = light_curve[clipped_indices]

        clipped_light_curve *= np.nan
        clipped_light_curve[clipped_indices] = light_curve
        ax[1].plot(clipped_light_curve[clipped_indices])

        # Reshape
        light_curve = light_curve.reshape(-1, 1)

        # Normalize flux values to standard range of [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1,1))
        # scaler = StandardScaler() # Standardize
        light_curve = scaler.fit_transform(light_curve)

        # Reshape
        light_curve = light_curve.flatten()

        # Replace NaN values with zeros
        clipped_light_curve[clipped_indices] = light_curve
        processed_light_curve[num_inds] = clipped_light_curve

        ax[2].plot(light_curve)
        fig.savefig('/home/echickle/foo.png')

        flux_data.append(processed_light_curve)

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

    pdb.set_trace()
    np.save(mydir+'sector-%02d_data.npy'%sector, padded_flux_data)

