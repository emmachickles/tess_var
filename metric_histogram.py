bls_dir = '/scratch/echickle/tess/BLS_results/'
plot_dir = '/scratch/echickle/tess/plots/'

sector_list = [61, 62, 63]
cam_list, ccd_list = [1,2,3,4], [1,2,3,4] 
# cam_list, ccd_list = [1], [1] 

# Import libraries
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs

tic_targ = [803489769, 36085812, 800042858, 270679102, 455206965, 452954413,
            767706310, 96852226, 677736827, 5393020, 101433897, 87550017, 
            826164830, 874726613, 193092806, 874383420, 192991819,
            808364853, 372519345, 272551828]

tic_list = []
sig_list = []
snr_list = []
wid_list = []
per_list = []
dur_list = []
rp_list = []
nt_list = []
dphi_list = []
mag_list = []


for sector in sector_list:
    for cam in cam_list:
        for ccd in ccd_list:

            # BLS statistics
            fname = bls_dir + 'BLS-{}-{}-{}.result'.format(sector, cam, ccd)
            res  = np.loadtxt(fname, delimiter=',')
            tic  = np.int64(res[:,0])
            sig  = res[:,3]
            snr  = res[:,4]
            wid  = res[:,5]
            per  = res[:,6]
            dur  = res[:,8]
            rp   = res[:,11]
            nt   = res[:,12]
            dphi = res[:,13]

            catalog_data = Catalogs.query_criteria(catalog='Tic', ID=tic)
            mag = np.array(catalog_data['Tmag'])

            tic_list.extend(tic)
            sig_list.extend(sig)
            snr_list.extend(snr)
            wid_list.extend(wid)
            per_list.extend(per)
            dur_list.extend(dur)
            rp_list.extend(rp)
            nt_list.extend(nt)
            dphi_list.extend(dphi)
            mag_list.extend(mag)

tic_list = np.array(tic_list)
sig_list = np.array(sig_list)
snr_list = np.array(snr_list)
wid_list = np.array(wid_list)
per_list = np.array(per_list)
dur_list = np.array(dur_list)
rp_list = np.array(rp_list)
nt_list = np.array(nt_list)
dphi_list = np.array(dphi_list)
mag_list = np.array(mag_list)

tic_targ = np.array(tic_targ)
_, comm1, comm2 = np.intersect1d(tic_targ, tic_list, return_indices=True)

tic_targ = tic_list[comm2]
sig_targ = sig_list[comm2]
snr_targ = snr_list[comm2]
wid_targ = wid_list[comm2]
per_targ = per_list[comm2]
dur_targ = dur_list[comm2]
rp_targ = rp_list[comm2]
nt_targ = nt_list[comm2]
dphi_targ = dphi_list[comm2]
mag_targ = mag_list[comm2]

bins = 50
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
# colors = plt.cm.viridis_r(range(len(tic_targ)))
colors = plt.get_cmap('viridis')(np.linspace(0.,1., len(tic_targ)))
for i in range(len(tic_targ)):
    label='TIC'+str(tic_targ[i])
    ax[0][0].axvline(sig_targ[i], ls='--', label=label, c=colors[i])
    ax[0][1].axvline(snr_targ[i], ls='--', label=label, c=colors[i])
    ax[0][2].axvline(wid_targ[i], ls='--', label=label, c=colors[i])
    ax[1][0].axvline(per_targ[i], ls='--', label=label, c=colors[i])
    ax[1][1].axvline(dur_targ[i], ls='--', label=label, c=colors[i])
    ax[1][2].axvline(rp_targ[i], ls='--', label=label, c=colors[i])
    ax[2][0].axvline(nt_targ[i], ls='--', label=label, c=colors[i])
    ax[2][1].axvline(dphi_targ[i], ls='--', label=label, c=colors[i])
    ax[2][2].axvline(mag_targ[i], ls='--', label=label, c=colors[i])

for a in ax.flatten():
    a.set_yscale('log')
    a.set_xscale('log')

ax[0][0].hist(sig_list, bins, alpha=0.5)
ax[0][0].set_xlabel('Peak significance')
ax[0][0].legend(fontsize='xx-small')

ax[0][1].hist(snr_list, bins, alpha=0.5)
ax[0][1].set_xlabel('Eclipse SNR')
# ax[0][1].legend(fontsize='x-small')

ax[0][2].hist(wid_list, bins, alpha=0.5)
ax[0][2].set_xlabel('Peak width')
# ax[0][2].legend(fontsize='x-small')

ax[1][0].hist(per_list, bins, alpha=0.5)
ax[1][0].set_xlabel('Period [days]')
ax[1][0].set_xscale('log')
# ax[1][0].legend(fontsize='x-small')

ax[1][1].hist(dur_list, bins, alpha=0.5)
ax[1][1].set_xlabel('Eclipse depth')
# ax[1][1].legend(fontsize='x-small')

ax[1][2].hist(rp_list, bins, alpha=0.5)
ax[1][2].set_xlabel('Companion radius [Earth radii]')
# ax[1][2].legend(fontsize='x-small')

ax[2][0].hist(nt_list, bins, alpha=0.5)
ax[2][0].set_xlabel('Number of in-eclipse points')
# ax[2][0].legend(fontsize='x-small')

ax[2][1].hist(dphi_list, bins, alpha=0.5)
ax[2][1].set_xlabel('Phase entropy')
ax[2][1].set_yscale('log')
# ax[2][1].legend(fontsize='x-small')

ax[2][2].hist(mag_list, bins, alpha=0.5)
ax[2][2].set_xlabel('TESS magnitude')
# ax[2][2].legend(fontsize='x-small')

plt.tight_layout()
plt.savefig(plot_dir+'cvae_hist_targets.png')
plt.close()
print('Saved '+plot_dir+'cvae_hist_targets.png')
