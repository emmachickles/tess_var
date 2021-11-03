# ==============================================================================
# ==============================================================================
# 
# feat_eng.py // Emma Chickles
# 
# -- Data Cleaning -- 
# * sigma_clip_data                     Remove outlier data pts and flares
#   * sigma_clip_lc
#   * sigma_clip_diag
# 
# -- Feature Generation --
# LS Periodograms
#   * compute_ls_pgram_data             LS-Periodograms from light curves
#     * compute_ls_pgram
#     * preprocess_lspgram              Prepare for ML
# Phase Curve Features
#   * create_phase_curve_feats
#     * mask_harmonics
#     * find_periodic_obj
#     * calc_phase_curve
# 
# ==============================================================================
# ==============================================================================

from __init__ import *
from scipy.stats import sigmaclip
from astropy.timeseries import LombScargle
from astropy.io import fits

# ==============================================================================
# ==============================================================================
# == Data Cleaning  ============================================================
# ==============================================================================
# ==============================================================================

# >> Sigma clipping
def sigma_clip_data(mg, plot=True, plot_int=200, n_sigma=7,
                    timescaldtrn=1/24., sectors=[]):
    '''Produces clip/ directory, which mirrors structure in raws/, and contains
    sigma clipped light curves.'''

    if len(sectors) == 0:
        sectors = os.listdir(mg.datapath+'mask/')
        sectors.sort()

    for sector in sectors: # >> loop through sectors
        sector_path = mg.datapath+'mask/'+sector+'/' # >> list of sectors
        lcfile_list = os.listdir(sector_path) # >> light curve file names
        clip_path = mg.datapath + 'clip/' + sector + '/' # >> output light curves
        dt.create_dir(clip_path)
        savepath = mg.savepath + 'clip/' + sector + '/' # >> output dir for plots
        dt.create_dir(savepath) 

        # >> loop through each light curve and sigma clip
        for i in range(len(lcfile_list)):
            lcfile = sector_path+lcfile_list[i]
            if i % plot_int == 0:
                verbose=True
                verbose_msg='Sigma clipping light curve '+sector+' '+\
                    str(i)+'/'+str(len(lcfile_list))
                if plot: plot_clip = True
            else:
                plot_clip, verbose = False, False

            # >> sigma clip light curve
            sigma_clip_lc(mg, lcfile, plot=plot_clip, n_sigma=n_sigma,
                          timescaldtrn=timescaldtrn, sector=sector)

        # if plot:
        #     clean_sector_diag(sector_path, savepath, sector, mdumpcsv)


def sigma_clip_lc(mg, lcfile, sector='', n_sigma=10, plot=False, max_iter=5,
                  timescaldtrn=1/24., savepath=None):

    if type(savepath) == type(None):
        savepath = mg.savepath + 'clip/' + sector + '/'

    # >> load light curve
    data, meta = dt.open_fits(fname=lcfile)
    time = data['TIME']
    flux = data['FLUX']

    # >> initialize variables 
    n_clip = 1 # >> number of data points clipped in an iteration
    n_iter = 0 # >> number of iterations
    flux_clip = np.copy(flux) # >> initialize sigma-clipped flux

    if plot:
        # title='n_sigma: {}, timescaldtrn: {:.2f} hr'.format(n_sigma,
        #                                                     timescaldtrn*24)
        title='n_sigma: '+str(n_sigma)+' timescaldtrn: '+\
            str(np.round(timescaldtrn*24, 3))+'hr'
        prefix='sigmaclip_'
        pt.plot_lc(time, flux, lcfile, prefix=prefix, title=title,
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
                              timescalbdtrspln=timescaldtrn)
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
            suffix='_niter'+str(n_iter)+'_dtrn'
            pt.plot_lc(time[num_indx], flux_dtrn, lcfile, title=title,
                       prefix=prefix, suffix=suffix, mdumpcsv=mg.mdumpcsv,
                       output_dir=savepath)

            suffix='_niter'+str(n_iter)
            pt.plot_lc(time, flux_clip, lcfile, prefix=prefix, title=title,
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
    print('Total NUM_CLIP: '+str(num_clip))
    table_meta = [('NUM_CLIP', num_clip)]
    ticid = meta['TICID']

    # >> save light curve in Fits file
    dt.write_fits(mg.datapath+'clip/'+sector+'/',
                  meta, [time, flux_clip],
                  ['TIME', 'FLUX'], table_meta=table_meta)

def sigma_clip_diag(mg, bins=40, cols=['Tmag', 'Teff'], n_div=6, ncols=2,
                    load_txt=True):
    '''Produces text file with TICIDs ranked by the number of data points masked
    during sigma clipping, and a histogram of those numbers.'''

    sectors = os.listdir(mg.datapath+'clip/')
    sectors.sort()
    savepath = mg.savepath + 'clip/hists/'
    dt.create_dir(savepath)


    # >> loop through sectors
    if load_txt:
        df = pd.read_csv(savepath+'sigmaclip.txt', sep='\t')
        sector_num = df['SECTOR']
        num_clip = df['NUM_CLIP']
        ticid = df['TICID']
        tess_feats = []
        for i in range(len(cols)):
            tess_feats.append(df[cols[i]])
        tess_feats = np.array(tess_feats).T
    else:

        # >> initialize arrays 
        ticid = np.empty(0)
        sector_num = np.empty(0)
        num_clip = np.empty(0)
        tess_feats = np.empty((0, len(cols)))

        for sector in sectors:
            print(sector)
            sector_path = mg.datapath+'clip/'+sector+'/' # >> list of sectors
            lcfile_list = os.listdir(sector_path) # >> light curve file names

            # >> read TESS features for sector
            tic_cat = pd.read_csv(mg.metapath+'spoc/tic/'+sector+'-tic_cat.csv',
                                  index_col=False)

            # >> loop through light curves in sector
            for lcfile in os.listdir(sector_path):
                objid = int(lcfile.split('.')[0])

                # >> open sigma-clipped light curve
                lchdu_clip = fits.open(sector_path+lcfile)
                objid = lchdu_clip[0].header['TICID']
                ticid = np.append(ticid, objid)
                sector_num = np.append(sector_num, int(sector.split('-')[1]))

                # >> open quality masked raw light curves (before clipping)
                lchdu_mask = fits.open(mg.datapath+'mask/'+sector+'/'+\
                                       str(int(objid))+'.fits')
                n = np.count_nonzero(np.isnan(lchdu_clip[1].data['FLUX'])) - \
                    np.count_nonzero(np.isnan(lchdu_mask[1].data['FLUX']))
                num_clip = np.append(num_clip, n)

                # >> find TESS features
                ind = np.nonzero(tic_cat['ID'].to_numpy() == objid)[0]
                if len(ind) == 0: # >> target not in tic_cat
                    target, feats = dt.get_tess_features(objid, cols=cols)
                else:
                    feats = []
                    for i in range(len(cols)):
                        feats.append(tic_cat.iloc[ind[0]][cols[i]])
                tess_feats = np.append(tess_feats, np.array([feats]), axis=0)

        # >> text file of number of clipped data points
        sort_indx = np.argsort(num_clip)[::-1]
        df_dict = {'TICID': ticid[sort_indx],
                   'SECTOR': sector_num[sort_indx],
                   'NUM_CLIP': num_clip[sort_indx]}
        for i in range(len(cols)):
            df_dict[cols[i]] = tess_feats[:,i]
        df = pd.DataFrame(df_dict)
        fname = savepath+'sigmaclip.txt'
        df.to_csv(fname, index=False, sep='\t')
        print('Saved '+fname)


    # >> number of clipped data points over all sectors
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(num_clip, bins=bins)
    ax.set_xlabel('Number of data points sigma-clipped')
    ax.set_ylabel('Number of targets')
    ax.set_yscale('log')
    # ax.set_title('Sector '+str(sector)+' Sigma Clipping')
    ax.set_title('Sigma Clipping: '+sectors[0]+' - '+sectors[-1])
    fname = savepath+'sigmaclip_hist.png'
    fig.tight_layout()
    fig.savefig(fname)
    print('Saved '+fname)

    # >> number of clipped points sorted by TESS features
    for i in range(len(cols)): # >> loop through TESS features
        sorted_inds = np.argsort(tess_feats[:,i])
        
        fig, ax = plt.subplots(nrows = int(n_div/ncols), ncols=ncols)

        # >> divide objects into div sections
        for row in range(int(n_div/ncols)):
            for col in range(ncols):
                a = ax[row, col]
                start = (row*ncols + col)*(len(tess_feats)//n_div)
                if row*ncols + col == n_div-1: # >> if last division
                    end = len(tess_feats) - 1
                else:
                    end = (row*ncols + col + 1)*(len(tess_feats)//n_div)
                # data = tess_feats[:,i][sorted_inds][start:end]
                data = num_clip[sorted_inds][start:end]
                feat_min = np.nanmin(tess_feats[:,i][sorted_inds][start:end])
                feat_max = np.nanmax(tess_feats[:,i][sorted_inds][start:end])
                title = str(np.round(feat_min,3)) + ' < '+cols[i]+' < '+\
                        str(np.round(feat_max,3))
                a.set_title(title)
                a.set_yscale('log')
                a.set_ylabel('Number of objects')
                a.set_xlabel('Number of clipped points')
                a.hist(data, bins=bins)

        fig.suptitle('Sigma Clipping: '+sectors[0]+' - '+sectors[-1])
        fig.tight_layout()
        fname = savepath+'sigmaclip_hist_'+cols[i]+'.png'
        fig.savefig(fname, dpi=300)
        print('Saved '+fname)

# ==============================================================================
# ==============================================================================
# == Feature Engineering =======================================================
# ==============================================================================
# ==============================================================================

# -- LS Periodograms -----------------------------------------------------------

def compute_ls_pgram_data(mg, plot=True, plot_int=200, n_freq=100000,
                          sectors=[]):
    """
    * mg : Mergen object """

    if len(sectors) == 0:
        sectors = os.listdir(mg.datapath+'clip/')
        sectors.sort()

    for sector in sectors:
        sector_path = mg.datapath+'clip/'+sector+'/' # >> list of sectors
        lcfile_list = os.listdir(sector_path) # >> light curve file names

        lspm_path = mg.datapath+'lspm/'+sector+'/'
        dt.create_dir(lspm_path)
        savepath = mg.savepath + 'lspm/'+sector+'/'
        dt.create_dir(savepath)

        # >> loop through each light curve and compute LS periodogram
        for i in range(len(lcfile_list)):
            lcfile = mg.datapath+'clip/'+sector+'/'+lcfile_list[i]

            if i % plot_int == 0:
                print('Computing LS periodogram of light curve '+sector+' '+\
                      str(i)+'/'+str(len(lcfile_list)))
                if plot: plot_pgram = True
            else:
                plot_pgram = False
            compute_ls_pgram(mg, lcfile, plot=plot_pgram, sector=sector,
                             n_freq=n_freq)

def compute_ls_pgram(mg, lcfile, plot=False, sector='', n_freq=200000,
                     max_freq=1/(8/1440.), min_freq=1/27., savepath=None,
                     verbose=False):
    '''min_freq : 8 minutes, > average Nyquist frequency
    max_freq : 27 days baseline
    '''

    if type(savepath) == type(None):
        savepath = mg.savepath + 'lspm/'+sector+'/'

    # >> open light curve file
    data, meta = dt.open_fits(fname=lcfile)
    time = data['TIME']
    flux = data['FLUX']
    ticid = meta['TICID']

    num_inds = np.nonzero(~np.isnan(flux))

    if type(n_freq) == type(None):
        freq, power = LombScargle(time[num_inds], flux[num_inds]).autopower()
    else:
        freq = np.linspace(min_freq, max_freq, n_freq)
        power = LombScargle(time[num_inds], flux[num_inds]).power(freq)
        

    # >> save periodogram
    lspm_path = mg.datapath+'lspm/'+sector+'/'
    dt.write_fits(lspm_path, meta, [freq, power], ['FREQ', 'LSPM'],
                  verbose=verbose)

    if plot:
        fig, ax = plt.subplots(3)
                
        lchdu_mask = fits.open(mg.datapath+'mask/'+sector+'/'+\
                               str(int(ticid))+'.fits')
        ax[0].set_title('Raw light curve (with quality flag mask)')
        ax[0].plot(lchdu_mask[1].data['TIME'], lchdu_mask[1].data['FLUX'],
                   '.k', ms=2, fillstyle='full')
        ax[0].set_xlabel('Time [BJD - 2457000]')
        ax[0].set_ylabel('Relative flux')

        ax[1].set_title('Sigma clipped light curve')
        ax[1].plot(time, flux, '.k', markersize=2, fillstyle='full')
        ax[1].set_xlabel('Time [BJD - 2457000]')
        ax[1].set_ylabel('Relative flux')

        ax[2].set_title('LS-periodogram')
        ax[2].plot(1/freq, power)
        ax[2].set_xlabel('Period (days)')
        ax[2].set_yscale('log')
        # ax[2].plot(freq, power)
        # ax[2].set_xlabel('Frequency (days -1)')
        ax[2].set_ylabel('LS Periodogram')
        # ax[i, 1].set_xscale('log')
        ax[1].set_yscale('log')
        fig.tight_layout()
        fname = savepath+'lspgram_TIC'+str(int(ticid))+'.png'
        fig.savefig(fname)
        print('Saved '+fname)
        plt.close(fig)


def preprocess_lspgram(mg, n_freq=100000):
    sectors = []
    for s in np.unique(mg.sectors):
        sectors.append('sector-%02d'%s)

    savepath = mg.datapath+'dae/'
    dt.create_dir(savepath)

    for sector in sectors:

        freq, lspgram, sector_num, ticid = [], [], [], []

        sector_path = mg.datapath+'lspm/'+sector+'/'
        lspmfile_list = os.listdir(sector_path)

        for i in range(len(lspmfile_list)): # >> open each LS periodogram
            if i % 200 == 0:
                print('Preprocessing LS periodogram '+sector+' '+str(i)+'/'+\
                      str(len(lspmfile_list)))
            lspmfile = mg.datapath+'lspm/'+sector+'/'+lspmfile_list[i]
            # start = datetime.now()
            with fits.open(lspmfile, memmap=False) as hdul:
                freq = hdul[1].data['FREQ']
                lspgram.append(hdul[1].data['LSPM'])
                sector_num.append(hdul[0].header['SECTOR'])
                ticid.append(hdul[0].header['TICID'])

            if lspgram[-1].shape[0] != n_freq:
                print(lspgram[-1].shape[0])
            if freq.shape[0] == 0 or lspgram[-1].shape[0] != n_freq:
                # >> retry opening the LSPM (object LS-periodogram file)
                with fits.open(lspmfile, memmap=False) as hdul:
                    freq = hdul[1].data['FREQ']
                    lspgram[-1] = hdul[1].data['LSPM']

            if i % 500 == 0: # >> prevents slowing
                gc.collect()
            # end = datetime.now()
            # dur_sec = (end-start).total_seconds()
            # print('Time to open Fits: '+str(dur_sec))

        # >> convert lists to numpy arrays
        freq, lspgram  = np.array(freq), np.array(np.stack(lspgram))
        sector_num, ticid = np.array(sector_num), np.array(ticid)

        # >> standardize
        lspgram = dt.standardize(lspgram, ax=1)

        # lspgram = lt.DAE_preprocessing(lspgram, norm_type='standardization', ax=1)

        # >> format 'P*()' creates column with variable length
        # >> format 'D' is floats, format 'K' is integers
        fname = sector+'-lspgram.fits'
        # dt.write_fits(savepath, None, [[lspgram, sector_num, ticid], [freq]],
        #               [['LSPM', 'SECTOR', 'TICID'], ['FREQ']],
        #               fmt=[['PD()', 'K', 'K'], ['D']], fname=fname,
        #               n_table_hdu=2)
        dt.write_fits(savepath, None, [[sector_num, ticid], [freq]],
                      [['SECTOR', 'TICID'], ['FREQ']],
                      fmt=[['K', 'K'], ['D']],
                      primary_data = lspm, fname=fname, n_table_hdu=2)

# def check_preprocess(mg, n_freq=100000):
#     sectors = []
#     for s in np.unique(mg.sectors):
#         sectors.append('sector-%02d'%s)
#     sectors.sort()
#     savepath = mg.datapath+'dae/'
#     for sector in sectors[1:]:
#         print(sector)
#         s_fname = savepath+sector+'-lspgram.fits' # >> sector file name
#         hdul = fits.open(s_fname)
#         shp = np.array([x.shape[0] for x in hdul[1].data['LSPM']])
#         inds = np.nonzero(shp != n_freq)
#         for i in inds[0]:
#             ticid = hdul[1].data['TICID'][i]
#             lspmfile = mg.datapath+'lspm/'+sector+'/'+str(ticid)+'.fits'
#             # >> replace LSPM for this TICID 
#             with fits.open(lspmfile, memmap=False) as l_hdu: # >> lspm hdu
#                 hdul[1].data['LSPM'][i] = l_hdu[1].data['LSPM']

#         dt.write_fits('', None, [[hdul[1].data['SECTOR'],
#                                         hdul[1].data['TICID']],
#                                        [hdul[2].data['FREQ']]],
#                       [['SECTOR', 'TICID'], ['FREQ']],
#                       fmt=[['K', 'K'], ['D']],
#                       primary_data = np.stack(hdul[1].data['LSPM']),
#                       fname=s_fname, n_table_hdu=2)
        
def load_lspgram(mg):
    lspmpath = mg.datapath+'dae/'

    fnames = []
    for s in np.unique(mg.sectors):
        fnames.append('sector-%02d'%s+'-lspgram.fits')

    freq, lspm, sector_num, ticid = [], [], [], []

    for fname in fnames:
        with fits.open(lspmpath+fname) as hdul:
            freq.append(hdul[2].data['FREQ'])
            lspm.append(hdul[0].data)
            sector_num.append(hdul[1].data['SECTOR'])
            ticid.append(hdul[1].data['TICID'])

    mg.freq = freq[0]
    mg.pgram = np.stack(np.concatenate(lspm, axis=0))
    mg.sectors = np.concatenate(sector_num)
    mg.objid = np.concatenate(ticid)

    # !!
    # inds = np.nonzero(np.array([len(l) for l in mg.pgram]) == 100000)
    # mg.pgram = np.array(mg.pgram[inds])
    # mg.sectors = mg.sectors[inds]
    # mg.objid = mg.objid[inds]
    # tmp = np.empty((len(mg.pgram), len(mg.pgram[0])))
    # for i in range(len(mg.pgram)):
    #     tmp[i] = mg.pgram[i]
    # mg.pgram=tmp
    
            
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
    # frequency = np.linspace(1./tmax, 1./tmin, n_freq)
    # power = LombScargle(t, y, nterms=n_terms).power(frequency)
    num_indx = np.nonzero(~np.isnan(y))
    t, y = t[num_indx], y[num_indx]
    frequency, power = LombScargle(t, y, nterms=n_terms).autopower()

    peak_c = np.argmax(power)
    peak_freq = np.linspace(frequency[peak_c-100], frequency[peak_c+100], 100000)
    pow_peak = LombScargle(t,y,nterms=n_terms).power(peak_freq)
    best_freq = peak_freq[np.argmax(pow_peak)]
    

    # best_freq = frequency[np.argmax(power)] 
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


