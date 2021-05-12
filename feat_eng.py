# ==============================================================================
# ==============================================================================
# 
# feat_eng.py // Emma Chickles
# 
# -- existing functions --------------------------------------------------------
# 
# -- TODO ----------------------------------------------------------------------
# 
# ==============================================================================
# ==============================================================================

from __init__ import *

def create_phase_curve_feats(time, flux, n_bins=100, n_freq=500000, 
                             n_terms=1, sector=1, output_dir='./'):

    feats = np.empty((len(flux), n_bins))

    # >> calculate phase curves
    for i in range(len(flux)):
        feats[i] = calc_phase_curve(time, flux[i], n_bins=n_bins, n_freq=n_freq,
                                    n_terms=n_terms)
        pdb.set_trace()

    feats = dt.standardize(feats)

    if not os.path.exists(output_dir+'pcfeat/'):
        os.makedirs(output_dir+'pcfeat/')
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(feats, header=hdr)
    hdu.writeto(output_dir+'pcfeat/Sector'+str(sector)+'-pcfeat.fits')

def calc_phase_curve(t, y, n_bins=100, n_freq=50000, n_terms=2,
                     har_window=100, kernel_size=25,
                     plot=False, output_dir='', prefix='', report_time=True,
                     n_freq0=10000, frac=20, frac_max=1e-3,
                     tmin=0.05, tmax=27, tmax0=5):

    # >> temporal baseline is 27 days, and shortest timescale sensitive to is
    # >> 4 minutes
    #tmax = 27 # days
    # tmin = 4./1440 # days
    #tmin = 0.025

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    import astropy.units as u

    if report_time:
        from datetime import datetime
        start = datetime.now()

    # -- check whether to do phase curve ---------------------------------------
    # >> first use sparse freq grid
    frequency = np.linspace(1./tmax0, 1./tmin, n_freq0)
    power = LombScargle(t, y, nterms=n_terms).power(frequency)    

    max_pow = np.max(power)
    max_ind = np.argmax(power)
    max_freq = frequency[max_ind] 
    
    # >> remove harmonics
    factors = np.arange(2,6)
    windows = []
    for factor in factors:
        har_ind = np.argmin(np.abs(frequency - max_freq/factor))
        # har_window = 300
        # har_window = har_wind * 10**frequency[har_ind]
        # har_window = har_wind * frequency[har_ind]
        window = np.arange(np.max([0,har_ind-har_window]),
                           np.min([har_ind+har_window, len(frequency)]),
                           dtype='int')
        windows.append(window)

        har_ind = np.argmin(np.abs(frequency - max_freq*factor))
                
        # har_window = har_wind * frequency[har_ind]
        window = np.arange(np.max([0,har_ind-har_window]),
                           np.min([har_ind+har_window, len(frequency)]),
                           dtype='int')
        windows.append(window)
    # >> frequency inds corresponding to harmonics
    inds = np.array([i for wind in windows for i in wind]).astype('int')

    # >> add max peak
    # max_window = max_wind * frequency[max_wind]
    # window = np.arange(np.max([0,max_ind-max_window]),
    #                    np.min([max_ind+max_window, len(frequency)]))

    # >> find width
    from scipy.signal import medfilt
    smoothed_power = medfilt(power, kernel_size=kernel_size)
    # no_peak_inds = np.nonzero((power-1e-3*np.max(power))<0)[0]
    no_peak_inds = np.nonzero((smoothed_power-frac_max*np.max(smoothed_power))<0)[0]
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
    # inds = np.append(inds, np.arange(0, np.argmin(frequency-mask_freq)))
    
    if plot:
        fig, ax = plt.subplots(2, figsize=(8, 2*3))
        ax[0].set_title('Sparse frequency grid (nfreq0='+str(n_freq0)+')') 
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
        fig.savefig(output_dir+prefix+'sparse_pgram.png')
        print('Saved '+output_dir+prefix+'sparse_pgram.png')
        plt.close(fig)

    # >> look for a peak that isn't a harmonic
    if np.max(np.delete(power, inds)) > max_pow/frac:
        print('Not making phase curve...')
        return
    else:

        # -- make phase curve --------------------------------------------------
        frequency = np.linspace(1./tmax, 1./tmin, n_freq)
        power = LombScargle(t, y, nterms=n_terms).power(frequency)

        best_freq = frequency[np.argmax(power)] 
        period = 1/best_freq

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
        pc_feats = np.array(np.split(ts_folded['flux'][:new_len], n_bins))
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
            print('Saved '+output_dir+prefix+'phase_curve.png')
            plt.close(fig)

        return pc_feats


