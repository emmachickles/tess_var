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
        # >> check if light curve is periodic
        res = mask_harmonics(time, flux[i], n_freq=n_freq0, n_terms=n_terms0,
                             report_time=report_time, plot=plot, output_dir=output_dir,
                             prefix='TIC'+str(ids[i]))

        feats = calc_phase_curve(time, flux[i], n_bins=n_bins, n_freq=n_freq,
                                 n_terms=n_terms, report_time=report_time, plot=plot,
                                 output_dir=output_dir, prefix='TIC'+str(ids[i]))
        with open(fname, 'a') as f:
            f.write(str(ids[i])+','+str(type(feats)==type(None)))
        if not type(feats) == type(None):
            all_feats.append(feats)

    feats = dt.standardize(feats)
    np.savetxt(output_dir+'Sector'+str(sector)+'_phase_curve_feat.txt',
               np.array(feats))

def mask_harmonics(t, y, n_freq=10000, n_terms=1, thresh_min=5e-3, thresh_max=5e-2,
                   tmin=0.05, tmax=5, har_window=100, kernel_size=25,
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
    
    if np.max(np.delete(power, inds)) > max_pow/thresh_max:
        print('Not making phase curve...')
        return False
    else:
        return True


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
        from datetime import datetime
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


