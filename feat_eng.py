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

def calc_phase_curve(t, y, n_bins=100, n_freq=500000, n_terms=2, har_window=100,
                     plot=False, output_dir='', prefix='', report_time=False):

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    import astropy.units as u

    if report_time:
        from datetime import datetime
        start = datetime.now()

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
        ax[1].set_xscale('log')
        ax[2].plot(ts_folded.time.value, ts_folded['flux'], '.k', ms=1)
        ax[2].set_xlabel('Time from midpoint epoch [days]')
        ax[2].set_ylabel('Relative Flux')
        ax[3].hist(pc_feats, bins=n_bins)
        ax[3].set_ylabel('Binned phase curve (nbins=' +str(n_bins)+ ')')
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'phase_curve.png')
        print('Saved '+output_dir+prefix+'phase_curve.png')
        plt.close(fig)

    return pc_feats


