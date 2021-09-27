# ==============================================================================
# ==============================================================================
# 
# feat_eng.py // Emma Chickles
# 
# -- existing functions --------------------------------------------------------
# 
# -- Preprocessing --
# * clean_sector
#   * clean_sector_diag
#   * sigma_clip
# 
# -- Feature Generation --
# LS Periodograms
#   * compute_ls_pgram_sector
#     * compute_ls_pgram
# Phase Curve Features
#   * create_phase_curve_feats
#     * mask_harmonics
#     * find_periodic_obj
#     * calc_phase_curve
# 
# -- TODO ----------------------------------------------------------------------
# 
# ==============================================================================

from __init__ import *
from scipy.stats import sigmaclip
from astropy.timeseries import LombScargle


# ==============================================================================
# ==============================================================================
# == Preprocessing =============================================================
# ==============================================================================
# ==============================================================================

# >> Sigma clipping
def sigma_clip_sector(mg, plot=True, plot_int=200, n_sigma=5):

    # >> get list of light curves names
    mask_sector_path = mg.datapath + 'mask/sector-%02d'%mg.sector+'/'
    lcfile_list = os.listdir(mask_sector_path)

    # >> create relevant directories
    clip_sector_path = mg.datapath + 'clip/sector-%02d'%mg.sector+'/'
    dt.create_dir(clip_sector_path)
    savepath = mg.ensbpath + 'clip/'
    dt.create_dir(savepath) 

    # >> loop through each light curve and sigma clip
    for i in range(len(lcfile_list)):
        lcfile = lcfile_list[i]
        if i % plot_int == 0:
            verbose=True
            verbose_msg='Sigma clipping light curve '+str(i)+'/'+\
                        str(len(lcfile_list))
            if plot: plot_pgram = True
        else:
            plot_pgram, verbose = False, False


        # >> sigma clip light curve
        sigma_clip_lc(mg.datapath, lcfile, mg.sector, 
                      mdumpcsv=mg.mdumpcsv, plot=plot, n_sigma=n_sigma,
                      savepath=savepath)

    if plot:
        clean_sector_diag(mask_sector_path, savepath, sector, mdumpcsv)


def sigma_clip_lc(mg,  n_sigma=5, plot=False, max_iter=5,
                  timescalbdtrspln=0.5/24.):

    mask_sector_path = mg.datapath + 'mask/sector-%02d'%mg.sector+'/'
    clip_sector_path = mg.datapath + 'clip/sector-%02d'%mg.sector+'/'
    savepath = mg.ensbpath + 'clip/'

    # >> load light curve
    time, flux, meta = dt.open_fits(mask_sector_path, fname=lcfile)

    # >> initialize variables 
    n_clip = 1 # >> number of data points clipped in an iteration
    n_iter = 0 # >> number of iterations
    flux_clip = np.copy(flux) # >> initialize sigma-clipped flux

    if plot:
        pt.plot_lc(time, flux, mask_sector_path+lcfile, prefix='sigmaclip_',
                   suffix='_niter0', mdumpcsv=mg.mdumpcsv, output_dir=savepath)

    while n_clip > 0:

        n_iter += 1

        # >> temporarily remove NaNs
        num_indx = np.nonzero(~np.isnan(flux_clip))
        flux_num = flux_clip[num_indx]

        # >> trial detrending
        # flux_dtrn = detrend(flux_num)
        flux_dtrn, _, _, _, _ = \
            ephesus.bdtr_tser(time[num_indx], flux_num,
                              timescalbdtrspln=timescalbdtrspln)
        flux_dtrn = np.concatenate(flux_dtrn)

        # >> sigma-clip light curve
        clipped, lower, upper = sigmaclip(flux_dtrn, low=n_sigma,
                                          high=n_sigma)
        
        # >> data points that are beyond threshold
        clip_indx = np.nonzero((flux_dtrn > upper) + (flux_dtrn < lower))[0]

        # >> apply sigma-clip mask
        flux_clip[num_indx[0][clip_indx]] = np.nan

        # >> plot the sigma-clipped time-series data
        if plot:
            prefix='sigmaclip_'
            suffix='_niter'+str(n_iter)+'_dtrn'
            pt.plot_lc(time[num_indx], flux_dtrn, mask_sector_path+lcfile,
                       prefix=prefix, suffix=suffix, mdumpcsv=mg.mdumpcsv,
                       output_dir=savepath)

            prefix='sigmaclip_'
            suffix='_niter'+str(n_iter)
            pt.plot_lc(time, flux_clip, mask_sector_path+lcfile, prefix=prefix,
                       mdumpcsv=mg.mdumpcsv, suffix=suffix, output_dir=savepath)

        # >> break loop if no data points are clipped
        n_clip = clip_indx.size
        print('Iteration number: '+str(n_iter)+', Num clipped points: '+\
              str(n_clip))

        # >> break loop if number of iterations exceeds limit
        if n_iter == 5:
            break

    # >> save number of clipped data points
    num_clip = np.count_nonzero(np.isnan(flux_clip)) - \
               np.count_nonzero(np.isnan(flux))
    table_meta = [('NUM_CLIP', num_clip)]
    ticid = meta['TICID']

    # >> save light curve in Fits file
    dt.write_fits(clip_sector_path, meta, ticid, [time, flux_clip],
                  ['TIME', 'FLUX'], table_meta=table_meta)

def sigma_clip_sector_diag(maskpath, savepath, sector, mdumpcsv, bins=40):
    '''Produces text file with TICIDs ranked by the number of data points masked
    during sigma clipping, and a histogram of those numbers.'''

    ticid = np.empty(0)
    num_clip = np.empty(0)
    for lcfile in os.listdir(maskpath):
        lchdu = fits.open(maskpath+lcfile)
        ticid = np.append(ticid, lchdu[0].header['TICID'])
        num_clip = np.append(num_clip, lchdu[1].header['NUM_CLIP'])

    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(num_clip, bins=bins)
    ax.set_xlabel('Number of data points sigma-clipped')
    ax.set_ylabel('Number of targets')
    ax.set_title('Sector '+str(sector)+' Sigma Clipping')
    fname = savepath+'Sector'+str(sector)+'_sigmaclip_hist.png'
    fig.tight_layout()
    fig.savefig(fname)
    print('Saved '+fname)

    sort_indx = np.argsort(num_clip)[::-1]
    df = pd.DataFrame({'TICID': ticid[sort_indx],
                       'NUM_CLIP': num_clip[sort_indx]})
    fname = savepath+'Sector'+str(sector)+'_sigmaclip.txt'
    df.to_csv(fname, index=False, sep='\t')
    print('Saved '+fname)



# ==============================================================================
# ==============================================================================
# == Feature Engineering =======================================================
# ==============================================================================
# ==============================================================================

# -- LS Periodograms -----------------------------------------------------------

def compute_ls_pgram_sector(mg, plot=True, plot_int=200):
    """
    * mg : Mergen object """

    mask_sector_path = mg.datapath + 'mask/sector-%02d'%mg.sector+'/'
    lspm_sector_path = mg.datapath + 'lspm/sector-%02d'%mg.sector+'/'
    dt.create_dir(lspm_sector_path)
    lcfile_list = os.listdir(mask_sector_path)

    savepath = mg.ensbpath + 'lspm/'
    dt.create_dir(savepath)

    for i in range(len(lcfile_list)):
        if i % plot_int == 0:
            print('Computing LS periodogram of light curve '+str(i)+'/'+\
                  str(len(lcfile_list)))
            if plot: plot_pgram = True
        else:
            plot_pgram = False
        compute_ls_pgram(mg, lcfile_list[i], plot=plot_pgram)

def compute_ls_pgram(mg, lcfile, plot=False):

    mask_sector_path = mg.datapath + 'mask/sector-%02d'%mg.sector+'/'
    lspm_sector_path = mg.datapath + 'lspm/sector-%02d'%mg.sector+'/'
    savepath = mg.ensbpath + 'lspm/'

    # >> open light curve file
    time, flux, meta = dt.open_fits(mask_sector_path, fname=lcfile)
    ticid = meta['TICID']

    num_inds = np.nonzero(~np.isnan(flux))
    freq, power = LombScargle(time[num_inds], flux[num_inds]).autopower()

    # >> save periodogram
    dt.write_fits(lspm_sector_path, meta, ticid, [freq, power], ['FREQ', 'LPSM'])

    if plot:
        fig, ax = plt.subplots(2)
        ax[0].plot(time, flux, '.k', markersize=2)
        ax[1].plot(freq, power)
        ax[0].set_xlabel('Time [BJD - 2457000]')
        ax[0].set_ylabel('Relative flux')
        ax[1].set_xlabel('Frequency (days -1)')
        ax[1].set_ylabel('LS Periodogram')
        # ax[i, 1].set_xscale('log')
        ax[1].set_yscale('log')
        fig.tight_layout()
        fname = savepath+'lspgram_TIC'+str(int(ticid))+'.png'
        fig.savefig(fname)
        print('Saved '+fname)

# -- Phase Curve Features ------------------------------------------------------

def create_phase_curve_feats(time, flux, ids, n_bins=100, n_freq=500000, 
                             n_terms=1, n_freq0=10000, n_terms0=1,
                             sector=1, output_dir='./', plot=False,
                             report_time=False):

    # >> textfile showing TICIDs where phase curve features were computed
    fname = output_dir+'Sector'+str(sector)+'_phase_curve_feat_gen.txt'
    with open(fname, 'w') as f: 
        f.write('')

    all_feats = []
    # >> calculate phase curves
    for i in range(len(flux)):

        if report_time: # >> restart timer
            start = datetime.now()

        # >> check if light curve is periodic
        res = mask_harmonics(time, flux[i], n_freq=n_freq0, n_terms=n_terms0,
                             report_time=report_time, plot=plot, output_dir=output_dir,
                             prefix='TIC'+str(ids[i]))

        # >> calculate phase curve features
        feats = calc_phase_curve(time, flux[i], n_bins=n_bins, n_freq=n_freq,
                                 n_terms=n_terms, report_time=report_time, plot=plot,
                                 output_dir=output_dir, prefix='TIC'+str(ids[i]))

        with open(fname, 'a') as f:
            f.write(str(ids[i])+','+str(type(feats)==type(None)))
        if not type(feats) == type(None):
            all_feats.append(feats)

        if report_time:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            print('Time to make phase curve: '+str(dur_sec))


    # feats = dt.standardize(feats, ax=0)
    all_feats = dt.standardize(all_feats, ax=0)
    np.savetxt(output_dir+'Sector'+str(sector)+'_phase_curve_feat.txt',
               np.array(all_feats))

def mask_harmonics(t, y, n_freq=10000, n_terms=1, thresh_min=5e-3, thresh_max=5e-2,
                   tmin=0.05, tmax=2, har_window=100, kernel_size=25,
                   report_time=True, plot=True, output_dir='', prefix=''):
    ''' Mask harmonics of the largest power to determine whether it is
    appropriate to compute phase curve features. Should take ~1 ms for short-
    cadence light curves.

    thresh_min : minimum threshold power for a peak (used to determine the width
                 of maximum peak)
    thresh_max : maximum threshold power for a peak (used to determine if there is
                 more than one signficiant component in periodogram'''

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    from scipy.signal import medfilt
    import astropy.units as u

    if report_time: # >> restart timer
        from datetime import datetime
        start = datetime.now()

    # -- calculate a sparse periodogram ----------------------------------------
    frequency = np.linspace(1./tmax, 1./tmin, n_freq)
    power = LombScargle(t, y, nterms=n_terms).power(frequency)    
    max_pow = np.max(power)
    max_ind = np.argmax(power)
    max_freq = frequency[max_ind] 
    
    # -- mask harmonics --------------------------------------------------------
    factors = np.arange(2,6)
    windows = []
    for factor in factors:
        har_ind = np.argmin(np.abs(frequency - max_freq/factor))
        window = np.arange(np.max([0,har_ind-har_window]),
                           np.min([har_ind+har_window, len(frequency)]),
                           dtype='int')
        windows.append(window)

        har_ind = np.argmin(np.abs(frequency - max_freq*factor))
                
        window = np.arange(np.max([0,har_ind-har_window]),
                           np.min([har_ind+har_window, len(frequency)]),
                           dtype='int')
        windows.append(window)

    # >> frequency inds corresponding to harmonics
    inds = np.array([i for wind in windows for i in wind]).astype('int')
    inds = np.unique(inds)

    # -- mask maximum peak -----------------------------------------------------
    # >> find width of maximum peak
    smoothed_power = medfilt(power, kernel_size=kernel_size)
    no_peak_inds = np.nonzero((smoothed_power-thresh_min*np.max(smoothed_power))<0)[0]
    sorted_inds = no_peak_inds[np.argsort(np.abs(no_peak_inds - max_ind))]
    if max_ind < np.min(sorted_inds):
        left_ind = 0
    else:
        left_ind = sorted_inds[np.nonzero(sorted_inds < max_ind)[0][0]]
    if max_ind > np.max(sorted_inds):
        right_ind = len(power)-1
    else:
         right_ind = sorted_inds[np.nonzero(sorted_inds > max_ind)[0][0]]
    window = np.arange(left_ind, right_ind)
    windows.append(window)
    inds = np.append(inds, window)

    # -- plot ------------------------------------------------------------------

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        print('Time to find second largest component: '+str(dur_sec))
    
    if plot:
        fig, ax = plt.subplots(2, figsize=(8, 2*3))
        ax[0].set_title('Sparse frequency grid (nfreq0='+str(n_freq)+')') 
        ax[0].plot(t, y, '.k', ms=1)
        ax[0].set_xlabel('Time [BJD - 2457000]')
        ax[0].set_ylabel('Relative Flux')
        ax[1].plot(frequency, power, '.k', ms=1)
        ax[1].set_xlabel('Frequency [1/days]')
        ax[1].set_ylabel('Power')

        # >> add second largest component to plotted windows
        inds = inds.astype('int')
        comp_ind = np.argmax(np.delete(power, inds))
        freq2 = np.delete(frequency, inds)[comp_ind]
        freq2_ind = np.argmin(np.abs(frequency - freq2))
        window = np.arange(np.max([0,har_ind-har_window]),
                           np.min([har_ind+har_window, len(frequency)]),
                           dtype='int')
        windows.append(window)
    
        # >> plot harmonics
        for i in range(len(windows)):
            window = windows[i]
            if i == len(windows)-2: # >> maximum peak
                c='m'
            elif i == len(windows)-1:
                c='r'
            else: # >> harmonics
                c='b'
            ax[1].axvspan(frequency[window[0]], frequency[window[-1]],
                          alpha=0.2, facecolor=c)
            freq = frequency[window[len(window)//2]]
            ax[1].axvline(freq, alpha=0.4, c=c)
            ax[1].text(freq, 0.85*np.max(power),str(np.round(1/freq,3))+'\ndays',
                       fontsize='small')
        
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'nfreq'+str(n_freq)+'-nterms'+\
                    str(n_terms)+'sparse_pgram.png')
        print('Saved '+output_dir+prefix+'sparse_pgram.png')
        plt.close(fig)

    # -- return result ---------------------------------------------------------
    
    if np.max(np.delete(power, inds)) > max_pow*thresh_max:
        print('Not making phase curve...')
        return False
    else:
        return True

def find_periodic_obj(ticid, targets, flux, time, sector, output_dir,
                      n_freq0=10000, report_time=False, plot=False,
                      thresh_min=1e-2):

    results = []
    # >> first find all the periodic objects
    fname = output_dir+'Sector'+str(sector)+'_periodic.txt'
    with open(fname, 'w') as f:
        f.write('TICID,PERIODIC\n')
        for t in targets:
            ind = np.nonzero(ticid==t)
            y = flux[ind][0]
            prefix = 'TIC'+str(t)

            res = mask_harmonics(time, flux[ind][0], n_freq=n_freq0,
                                 report_time=report_time, plot=plot,
                                 output_dir=output_dir, prefix='TIC'+str(t),
                                 thresh_min=thresh_min)
            f.write(str(t)+','+str(res)+'\n')
            results.append(res)
            print('TIC'+str(t)+' '+str(res))
    inds = np.nonzero(np.array(results))
    return targets[inds]

def calc_phase_curve(t, y, n_bins=100, n_freq=50000, n_terms=2,
                     plot=False, output_dir='', prefix='', report_time=True,
                     tmin=0.05, tmax=27):
    '''frac_max : adjusts the threshold power to determine the width of maximum peak
    frac : adjusts the threshold power to determine the signfiicance of second
    largest peak'''

    # >> temporal baseline is 27 days, and shortest timescale sensitive to is
    # >> 4 minutes
    #tmax = 27 # days
    # tmin = 4./1440 # days
    #tmin = 0.025

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    import astropy.units as u

    if report_time: # >> restart timer
        start = datetime.now()

    # -- compute periodogram ---------------------------------------------------
    frequency = np.linspace(1./tmax, 1./tmin, n_freq)
    power = LombScargle(t, y, nterms=n_terms).power(frequency)

    best_freq = frequency[np.argmax(power)] 
    period = 1/best_freq

    # -- compute phase curve ---------------------------------------------------
    time = Time(t, format='jd')
    ts = TimeSeries(time=time, data={'flux': y})
    ts_folded = ts.fold(period=period*u.d) 

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        print('Time to make phase curve: '+str(dur_sec))

    # >> bin phase curve
    orig_len = ts_folded['flux'].shape[0]
    new_len = orig_len - orig_len%n_bins

    folded = ts_folded['flux'][np.argsort(ts_folded.time.value)]
    pc_feats = np.array(np.split(folded[:new_len], n_bins))
    pc_feats = np.mean(pc_feats, axis=1)

    if plot:
        fig, ax = plt.subplots(4, figsize=(8, 4*3))
        if report_time:
            ax[0].set_title('Computation time: '+str(dur_sec)) 
        ax[0].plot(t, y, '.k', ms=1)
        ax[0].set_xlabel('Time [BJD - 2457000]')
        ax[0].set_ylabel('Relative Flux')
        ax[1].set_title('Period: '+str(np.round(period, 3))+' days')
        ax[1].plot(frequency, power, '.k', ms=1)
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Power')
        # ax[1].set_xscale('log')
        ax[2].plot(ts_folded.time.value, ts_folded['flux'], '.k', ms=1)
        ax[2].set_xlabel('Time from midpoint epoch [days]')
        ax[2].set_ylabel('Relative Flux')
        ax[3].plot(np.arange(len(pc_feats)), pc_feats, '.')
        ax[3].set_ylabel('Binned phase curve (nbins=' +str(n_bins)+ ')')
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'phase_curve.png')
        print('Saved '+output_dir+prefix+'nfreq'+str(n_freq)+'-nterms'+\
              str(n_terms)+'-phase_curve.png')
        plt.close(fig)

    return pc_feats


