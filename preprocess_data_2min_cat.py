import pdb
import os
import numpy as np
import pandas as pd

# Path to the directory containing quality flag masked light curves
lc_dir = '/scratch/data/tess/lcur/spoc/mask/'

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('/scratch/echickle/hlsp_tess-svc_tess_lcf_all-s0001-s0026_tess_v1.0_cat.csv')

binned_LC_list = []
ticid_list = []

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    if index%100==0:
        print(index)
    tess_id = row['tess_id']
    sector = row['Sector']
    if sector[0] == 's':
        sector = sector[2:]
    period_var = row['period_var']
    bjd0_var = row['bjd0_var']
    flux_file_path = os.path.join(lc_dir, 'sector-%02d'%np.int64(sector), f'{tess_id}.npy')

    # Load the flux values from the .npy file
    flux_data = np.load(flux_file_path)

    # Extract time and flux from the loaded data
    time=flux_data[0]
    flux=flux_data[1]
    inds=np.nonzero( (~np.isnan(flux)) * (~np.isnan(time)) )
    time=time[inds]
    flux=flux[inds]

    # Phase fold 
    time=time-bjd0_var
    phases=(time % period_var)/period_var

    # Bin to 128 bins
    n_bins=128
    binned_LC=[]
    mean_phases=np.linspace(0,1-1/n_bins,n_bins)
    lightcurve=np.array((phases,flux)).T
    for i, phase in enumerate(mean_phases):
        lightcurve_bin=lightcurve[lightcurve[:,0]>phase]
        if i < len(mean_phases)-1:
            lightcurve_bin=lightcurve_bin[lightcurve_bin[:,0]<mean_phases[i+1]]
        mean_flux=np.mean(lightcurve_bin[:,1])
        mean_flux_error=np.std(lightcurve_bin[:,1])
        binned_LC.append((i+0.5/n_bins,mean_flux,mean_flux_error))
    binned_LC=np.array(binned_LC)

    # Interpolate 
    inds=np.isnan(binned_LC[:,1])
    binned_LC[:,1][inds]=np.interp(mean_phases[inds], mean_phases[~inds],
                             binned_LC[:,1][~inds])
    binned_LC[:,2][inds]=np.mean(binned_LC[:,2][~inds])

    binned_LC[:,2]=binned_LC[:,2]/np.nanmean(binned_LC[:,1])
    binned_LC[:,1]=binned_LC[:,1]/np.nanmean(binned_LC[:,1])
    binned_LC_list.append(binned_LC)
    ticid_list.append(tess_id)

    # 3 cycles
    # binned_LC2=np.array((binned_LC[:,0]-1,binned_LC[:,1],binned_LC[:,2])).T
    # binned_LC3=np.array((binned_LC[:,0]+1,binned_LC[:,1],binned_LC[:,2])).T
    # binned_LC=np.vstack((binned_LC2,binned_LC))
    # binned_LC=np.vstack((binned_LC,binned_LC3))

binned_LC_list = np.array(binned_LC_list)
fname = '/scratch/echickle/tess_binned_s0001-s0026_phase_curves.npy'
np.save(fname, binned_LC_list)

ticid_list = np.array(ticid_list)
fname = '/scratch/echickle/tess_binned_s0001-s0026_ticid.npy'
np.save(fname, ticid_list)

import pdb
pdb.set_trace()
