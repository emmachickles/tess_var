# conda environment: ml

import os
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import median_filter
from wotan import flatten
import sys
sys.path.insert(0, '/home/echickle/work/tess_dwd/')
import lc_utils as lcu

mydir = '/scratch/data/tess/lcur/ffi/cvae_data/'
os.makedirs(mydir, exist_ok=True)
qflag_dir = "/scratch/echickle/QLPqflags/"

sector_list = [56]
cam_list, ccd_list = [1,2,3,4], [1,2,3,4] # !!



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
    for cam in cam_list:
        for ccd in ccd_list:
            data_dir = '/scratch/data/tess/lcur/ffi/s%04d-lc/'%sector
            fname = data_dir+'ts-{}-{}.npy'.format(cam, ccd)
            
            # remove quality flags
            sector_dir = qflag_dir + 'sec%d/' % sector
            cn = np.load(data_dir+'cn-{}-{}.npy'.format(cam,ccd))
            file_names = os.listdir(sector_dir)
            file_names = [f for f in file_names if 'cam%dccd%d'%(cam, ccd) in f]
            qflag_data = []
            for f in file_names:
                qflag_data.extend(np.loadtxt(sector_dir+f))
            qflag_data = np.array(qflag_data)
            bad_inds = np.nonzero(qflag_data[:,1])[0]
            bad_cadence = qflag_data[:,0][bad_inds]
            _, comm1, comm2 = np.intersect1d(cn, bad_cadence, return_indices=True)
            cn = np.delete(cn, comm1)

            time_array = np.load(fname)
            time_array = np.delete(time_array, comm1)
            length_array.append(len(time_array))
max_time_length = max(length_array)

# Determine the desired length based on the model architecture
# NEED +1 FOR dense_length during preprocessing only
dense_length = max_time_length // (4 ** num_conv_layers) + 1 # >> length before Dense(latent_dim)
desired_length = dense_length * (4 ** num_conv_layers)

# Only process periodic sources
ticid_periodic = np.int64(np.loadtxt(mydir+'metric_cut.txt'))

# Load and preprocess light curve data
for sector in sector_list:
    flux_data = []
    ticid_array = []

    for cam in cam_list:
        for ccd in ccd_list:
            print('Sector {} Cam {} CCD {}'.format(sector, cam, ccd))

            # Directory containing the light curve data
            data_dir = '/scratch/data/tess/lcur/ffi/s%04d-lc/'%sector
            file_name = data_dir+'lc-{}-{}.npy'.format(cam, ccd)
            light_curve_array = np.load(file_name)

            # Only process periodic sources
            ticid = np.int64(np.load(data_dir+'id-{}-{}.npy'.format(cam,ccd)))
            _, inds, _ = np.intersect1d(ticid, ticid_periodic, return_indices=True)
            light_curve_array = light_curve_array[inds]
            ticid = ticid[inds]
            t= np.load(data_dir+'ts-{}-{}.npy'.format(cam,ccd))

            # >> remove nonzero quality flags
            sector_dir = qflag_dir + 'sec%d/' % sector
            cn = np.load(data_dir+'cn-{}-{}.npy'.format(cam,ccd))
            file_names = os.listdir(sector_dir)
            file_names = [f for f in file_names if 'cam%dccd%d'%(cam, ccd) in f]
            qflag_data = []
            for f in file_names:
                qflag_data.extend(np.loadtxt(sector_dir+f))
            qflag_data = np.array(qflag_data)
            bad_inds = np.nonzero(qflag_data[:,1])[0]
            bad_cadence = qflag_data[:,0][bad_inds]
            _, comm1, comm2 = np.intersect1d(cn, bad_cadence, return_indices=True)
            cn = np.delete(cn, comm1)
            t = np.delete(t, comm1)
            light_curve_array = np.delete(light_curve_array, comm1, axis=1)


            for i, light_curve in enumerate(light_curve_array):


                # Remove nan values
                # processed_light_curve = np.zeros_like(light_curve)
                num_inds = np.where(~np.isnan(light_curve))[0] 
                light_curve = light_curve[num_inds]
                t_clipped = t[num_inds]

                # Detrend
                light_curve, _ = lcu.normalize_lc(light_curve)
                light_curve = flatten(t_clipped, light_curve, window_length=0.1, method='biweight')
                inds = np.nonzero(~np.isnan(light_curve))
                t_clipped, light_curve = t_clipped[inds], light_curve[inds]

                # # Apply rolling median filter to the flux data
                # rolling_median = median_filter(light_curve, size=window_size, mode='reflect')

                # # Calculate the rolling standard deviation
                # rolling_std = np.std(light_curve - rolling_median)

                # # Define the upper and lower clip thresholds
                # upper_threshold = rolling_median + sigma_threshold * rolling_std
                # lower_threshold = rolling_median - sigma_threshold * rolling_std

                # # Perform sigma clipping
                # clipped_indices = np.where((light_curve < upper_threshold) & (light_curve > lower_threshold))[0]

                med = np.median(light_curve)
                std = np.std(light_curve)
                clipped_indices = np.where((light_curve>med-std*sigma_threshold) & (light_curve<med+std*sigma_threshold))

                # Return the clipped flux array
                # clipped_light_curve = np.zeros_like(light_curve)
                light_curve = light_curve[clipped_indices]
                t_clipped = t_clipped[clipped_indices]

                # Interpolate
                light_curve = np.interp(t, t_clipped, light_curve)

                # Reshape
                light_curve = light_curve.reshape(-1, 1)

                # Normalize flux values to standard range of [-1, 1]
                scaler = MinMaxScaler(feature_range=(-1,1))
                # scaler = StandardScaler() # Standardize
                light_curve = scaler.fit_transform(light_curve)

                # Reshape
                light_curve = light_curve.flatten()

                # # Replace NaN values with zeros
                # clipped_light_curve[clipped_indices] = light_curve
                # processed_light_curve[num_inds] = clipped_light_curve

                # flux_data.append(processed_light_curve)
                flux_data.append(light_curve)

                if np.std(light_curve) < 0.1:
                    pdb.set_trace()

            ticid_array.extend(ticid)
            pdb.set_trace() 

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

    np.save(mydir+'sector-%02d_data.npy'%sector, padded_flux_data)
    np.save(mydir+'sector-%02d_ticid.npy'%sector, ticid_array)
