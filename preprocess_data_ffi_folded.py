ztf = True

if ztf:
    mydir = '/scratch/data/tess/lcur/ffi/cvae_ztf_data/'
else:
    mydir = '/scratch/data/tess/lcur/ffi/cvae_data/'
bls_dir = '/scratch/echickle/tess/BLS_results/'
if ztf:
    plot_dir = '/scratch/echickle/tess/plots_ztf/'
else:
    plot_dir = '/scratch/echickle/tess/plots/'
qflag_dir = '/scratch/echickle/QLPqflags/'

N = 128
sector_list = [61]
# cam_list, ccd_list = [1,2,3,4], [1,2,3,4] 
cam_list, ccd_list = [1], [1, 2,3,4] 

# Variability thresholds
sig_min  = 15
wid_min  = 6
nt_min   = 30
dphi_max = 0.04
per_max  = 1

# Import libraries
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from wotan import flatten

# os.system('rm '+plot_dir+'cvae_instances/*.png')

for sector in sector_list:
    for cam in cam_list:
        for ccd in ccd_list:

            LC_list = []

            # BLS statistics
            fname = bls_dir + 'BLS-{}-{}-{}.result'.format(sector, cam, ccd)
            res  = np.loadtxt(fname, delimiter=',')
            tic  = np.int64(res[:,0])
            sig  = res[:,3]
            snr  = res[:,4]
            wid  = res[:,5]
            per  = res[:,6]
            dur  = res[:,8]
            epo  = res[:,10]
            rp   = res[:,11]
            nt   = res[:,12]
            dphi = res[:,13]

            bins = 100
            fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            ax[0][0].hist(sig, bins, alpha=0.5)
            ax[0][0].set_xlabel('Peak significance')
            ax[0][0].set_yscale('log')
            ax[0][1].hist(snr, bins, alpha=0.5)
            ax[0][1].set_xlabel('Eclipse SNR')
            ax[0][1].set_yscale('log')
            ax[0][2].hist(wid, bins, alpha=0.5)
            ax[0][2].set_xlabel('Peak width')
            ax[0][2].set_yscale('log')
            ax[1][0].hist(per, bins, alpha=0.5)
            ax[1][0].set_xlabel('Period [days]')
            ax[1][0].set_yscale('log')
            ax[1][1].hist(dur, bins, alpha=0.5)
            ax[1][1].set_xlabel('Eclipse depth')
            ax[1][1].set_yscale('log')
            ax[1][2].hist(rp, bins, alpha=0.5)
            ax[1][2].set_xlabel('Companion radius [Earth radii]')
            ax[1][2].set_yscale('log')
            ax[2][0].hist(nt, bins, alpha=0.5)
            ax[2][0].set_xlabel('Number of in-eclipse points')
            ax[2][0].set_yscale('log')
            ax[2][1].hist(dphi, bins, alpha=0.5)
            ax[2][1].set_xlabel('Phase entropy')
            ax[2][1].set_yscale('log')

            orig_num = len(tic)
            good_idx = np.nonzero( (sig  > sig_min) * \
                                   (wid  > wid_min) * \
                                   (nt   > nt_min)  * \
                                   (dphi < dphi_max) * \
                                   (per  < per_max) )
            tic  = tic[good_idx]
            sig  = sig[good_idx]
            snr  = snr[good_idx]
            wid  = wid[good_idx]
            per  = per[good_idx]
            dur  = dur[good_idx]
            epo  = epo[good_idx]
            rp   = rp[good_idx]
            nt   = nt[good_idx]
            dphi = dphi[good_idx]

            cut_num = len(tic)
            ax[0][1].set_title(str(cut_num) + ' / ' + str(orig_num))
            ax[0][0].hist(sig, bins, alpha=0.5)
            ax[0][1].hist(snr, bins, alpha=0.5)
            ax[0][2].hist(wid, bins, alpha=0.5)
            ax[1][0].hist(per, bins, alpha=0.5)
            ax[1][1].hist(dur, bins, alpha=0.5)
            ax[1][2].hist(rp, bins, alpha=0.5)
            ax[2][0].hist(nt, bins, alpha=0.5)
            ax[2][1].hist(dphi, bins, alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_dir+'cvae_hist.png')
            plt.close()

            # Light curve data
            data_dir = '/scratch/data/tess/lcur/ffi/s%04d-lc/'%sector
            time   = np.load(data_dir+'ts-{}-{}.npy'.format(cam, ccd))
            flux   = np.load(data_dir+'lc-{}-{}.npy'.format(cam, ccd))
            cn     = np.load(data_dir+'cn-{}-{}.npy'.format(cam, ccd))
            tic_lc = np.int64(np.load(data_dir+'id-{}-{}.npy'.format(cam, ccd)))

            # Only process instances that pass metric cut
            _, comm1, comm2 = np.intersect1d(tic_lc, tic, return_indices=True)
            tic_lc = tic_lc[comm1]
            flux   = flux[comm1]
            sig    = sig[comm2]
            wid    = wid[comm2]
            per    = per[comm2]
            dur    = dur[comm2]
            epo    = epo[comm2]
            rp     = rp[comm2]
            nt     = nt[comm2]
            dphi   = dphi[comm2]

            # Sort time array
            inds = np.argsort(time)
            time = time[inds]
            cn   = cn[inds]
            flux = flux[:,inds]
                            
            # Remove nonzero quality flags
            sector_dir = qflag_dir + 'sec%d/' % sector
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
            time = np.delete(time, comm1)
            flux = np.delete(flux, comm1, axis=1)
    
            # # !! 
            # ind = np.nonzero(tic_lc == 36085812)
            # for i in [ind[0][0]]:

            for i in range(len(flux)):
                lightcurve = np.array([time, flux[i]]).T

                # Remove nans
                inds = np.nonzero(~np.isnan(lightcurve[:,1]))
                lightcurve = lightcurve[inds]

                # Detrend
                if np.min(lightcurve[:,1]) < 0:
                    lightcurve[:,1] -= np.min(lightcurve[:,1])
                lightcurve[:,1] = flatten(lightcurve[:,0], lightcurve[:,1],
                                          window_length=0.1, method='biweight')
                inds = np.nonzero(~np.isnan(lightcurve[:,1]))
                lightcurve = lightcurve[inds]
                
                # Sigma clip
                n_std = 5
                med = np.median(lightcurve)
                std = np.std(lightcurve)
                inds = np.nonzero( (lightcurve[:,1] > med - n_std*std) * \
                                   (lightcurve[:,1] < med + n_std*std) )
                lightcurve = lightcurve[inds]

                # Phase
                mean_phi = np.linspace(0, 1-1/N, N)
                lightcurve[:,0] = ( (lightcurve[:,0] - epo[i]) % per[i] ) / per[i]
                inds = np.argsort(lightcurve[:,0])
                lightcurve = lightcurve[inds]
                binned_LC = []
                for phi in mean_phi:
                    lightcurve_bin = lightcurve[lightcurve[:,0] > phi]
                    lightcurve_bin = lightcurve_bin[lightcurve_bin[:,0] < phi+1/N]
                    binned_LC.append( (phi+0.5/N, np.mean(lightcurve_bin[:,1])) )
                binned_LC = np.array(binned_LC)
                # if np.count_nonzero(~np.isnan(binned_LC[:,1])) == 0: pdb.set_trace()

                binned_LC[:,1] = binned_LC[:,1] / np.nanmedian(binned_LC[:,1])
                LC_list.append(binned_LC[:,1])

                binned_LC2 = np.array( (binned_LC[:,0]-1, binned_LC[:,1]) ).T
                binned_LC3 = np.array( (binned_LC[:,0]+1, binned_LC[:,1]) ).T
                binned_LC = np.vstack( (binned_LC2, binned_LC) )
                binned_LC = np.vstack( (binned_LC, binned_LC3) )

                plt.figure(figsize=(10,5))
                plt.errorbar(binned_LC[:,0], binned_LC[:,1], np.ones(len(binned_LC))*0.01,
                             color='k', ms=2, ls=' ', elinewidth=1, capsize=2)
                plt.xlabel('Phases')
                plt.ylabel('Relative flux')
                plt.savefig(plot_dir+'cvae_instances/TIC{}.png'.format(tic_lc[i]))
                plt.close()

                # pdb.set_trace()

            LC_list = np.array(LC_list)
            np.save(mydir+'s%04d'%sector+'-cam{}-ccd{}.npy'.format(cam, ccd), LC_list)
            data = np.array([tic_lc, sig, wid, per, dur, rp, nt, dphi]).T
            np.save(mydir+'s%04d'%sector+'-cam{}-ccd{}-stats.npy'.format(cam, ccd), data)
            print('Finished S {} Cam {} CCD {}'.format(sector,cam,ccd))

                    


