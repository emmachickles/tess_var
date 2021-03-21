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

def phase_curve(t,y,output_dir='', prefix='', n_bins=100, plot=False,
                n_terms=2, har_window=100, n_freq=500000):

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    import astropy.units as u
    from datetime import datetime

    start = datetime.now()
    # frequency, power = LombScargle(t,y).autopower()

    # >> temporal baseline is 27 days, and shortest timescale sensitive to is
    # >> 4 minutes
    frequency = np.linspace(1./27, 1./(4/1440), n_freq)
    power = LombScargle(t, y, nterms=n_terms).power(frequency)

    best_freq = frequency[np.argmax(power)] 
    period = 1/best_freq

    # >> check if this is the fundamental frequency
    # factors = [2,3,5]
    # for factor in factors:
    #     ind = np.argmin(np.abs(frequency - best_freq/factor))
    #     # >> is there a peak at best_freq/factor
    #     window = power[np.max([0,ind-har_window]):ind+har_window]
    #     if np.max(window) > np.max(power)/2:
    #         print('Max frequency was a harmonic!')
    #         best_freq = frequency[ind-har_window+np.argmax(window)]
    
    time = Time(t, format='jd')
    ts = TimeSeries(time=time, data={'flux': y})
    ts_folded = ts.fold(period=period*u.d) 

    end = datetime.now()
    dur_sec = (end-start).total_seconds()
    print('Time to make phase curve: '+str(dur_sec))
    # >> bin phase curve
    # pc_feats = np.split(ts_folded['flux'], n_bins)
    # pc_feats = np.mean(pc_feats, axis=0)
    # pdb.set_trace()

    if plot:
        fig, ax = plt.subplots(3, figsize=(8, 3*3))
        ax[0].set_title('Computation time: '+str(dur_sec)) 
        ax[0].plot(t, y, '.k', ms=1)
        ax[0].set_xlabel('Time [BJD - 2457000]')
        ax[0].set_ylabel('Relative Flux')
        ax[1].set_title('Period: '+str(np.round(period, 3))+' days')
        ax[1].plot(frequency, power, '.k', ms=1)
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Power')
        ax[1].set_xscale('log')
        ax[2].plot(ts_folded.time.value, ts_folded['flux'], '.k', ms=1)
        ax[2].set_xlabel('Time from midpoint epoch [days]')
        ax[2].set_ylabel('Relative Flux')
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'phase_curve.png')
        print('Saved '+output_dir+prefix+'phase_curve.png')
        plt.close(fig)



