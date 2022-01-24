# ==============================================================================
# ==============================================================================
# 
# feat_eng.py // Emma Chickles
# 
# -- Data Cleaning -- 
# * sigma_clip_data                     Remove outlier data pts and flares
#   * sigma_clip_lc
#   * sigma_clip_diag
# * progress_checker
# * get_lc
# * plot_lspm
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
#   * simulated_data
#
#  Light curve and LS periodogram statistics
#   * compute_stats
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

# -- Sigma clipping ------------------------------------------------------------
def sigma_clip_data(mg, plot=True, plot_int=200, n_sigma=7,
                    timescaldtrn=1/24., sectors=[]):
    '''Produces mg.datapath/clip/ directory, which mirrors structure in 
    mg.datapath/raws/, and contains sigma clipped light curves.'''

    if len(sectors) == 0:
        sectors = os.listdir(mg.datapath+'mask/')
        sectors.sort()

    for sector in sectors: # >> loop through sectors

        fname = mg.metapath+'spoc/targ/2m/all_targets_S%03d'\
                %int(sector.split('-')[-1])+'_v1.txt'
        sector_ticid = np.loadtxt(fname)[:,0]

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

            ticid = int(lcfile.split('/')[-1].split('.')[0])
            if ticid in sector_ticid:
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

def progress_checker(path, old_date=None, rmv_old=False):
    import os 
    import time
    import pdb
    import numpy as np
    fnames = [path+f for f in os.listdir(path)]
    fnames.sort(key=lambda x: os.path.getmtime(x))
    times = [time.ctime(os.path.getmtime(x)) for x in fnames]
    # print(times)
    dates = np.unique([d[4:10] for d in times])
    for d in dates:
        print(d+': '+str(len([t for t in times if d in t])))

    if type(old_date) != type(None):
        date = [t for t in times if old_date in t][0]
        ind = times.index(date)
        if rmv_old:
            for i in range(ind + 1):
                os.remove(fnames[i])

        return fnames[ind:]

def get_lc(lcpath, ticid, timescale, norm=False, method='median', rmv_nan=False,
           detrend=False, plot=False, savepath=None, return_sector=False):

    # -- load raw light curve data ---------------------------------------------

    fnames = []
    for s in os.listdir(lcpath): # >> loop through sectors
        fnames.extend([lcpath+s+'/'+f for f in \
                       os.listdir(lcpath+s) \
                       if str(ticid) in f])
    t, y = [], []
    for fname in fnames:
        data, meta = dt.open_fits(fname=fname)  
        if rmv_nan:
            inds = np.nonzero(~np.isnan(data['FLUX']))
            t.append(data['TIME'][inds])
            y.append(data['FLUX'][inds])
        else:
            t.append(data['TIME'])
            y.append(data['FLUX'])
    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_lc(ax, np.concatenate(t), np.concatenate(y), c='k', ms=2,
                label='raw')

    if norm: # -- normalization ------------------------------------------------
        for i in range(len(y)):
            y[i] = dt.normalize(y[i], axis=0, method=method)

        if plot:
            plot_lc(ax, np.concatenate(t), np.concatenate(y), c='r', ms=2,
                    label='normalized')            

    if detrend: # -- linearly detrending ---------------------------------------
        for i in range(len(y)):
            y[i] = detrend_lc(t[i], y[i])
        if plot:
            plot_lc(ax, np.concatenate(t), np.concatenate(y),
                    label='detrend', ms=2)

    # -- concatenate -----------------------------------------------------------

    if plot:
        fig.tight_layout()
        out_f = savepath+'lightcurve_TIC'+str(ticid)+'.png'
        fig.savefig(out_f)
        print('Saved '+out_f)

    if timescale == 1:
        if return_sector:
            return t, y
        t, y = np.concatenate(t), np.concatenate(y)
    else:
        t = np.concatenate(t[:timescale])
        y = np.concatenate(y[:timescale])

    if rmv_nan:
        inds = np.nonzero(~np.isnan(y))
        t=t[inds]
        y=y[inds]

    return t, y

# ==============================================================================
# ==============================================================================
# == Feature Engineering =======================================================
# ==============================================================================
# ==============================================================================

# -- LS Periodograms -----------------------------------------------------------

def calc_lspgram_sing_sector(mg, plot=True, plot_int=200, sectors=[],
                             n0=6, overwrite=True, report_time=True):
    """Compute LS-periodograms with a baseline of a single observational sector 
    (~27 days). Periodograms for stars that are observed in multiple sectors are
    averaged for each frequency bin.
    * mg : Mergen object
    * n0 : oversampling factor
    """

    # -- get light curve file names --------------------------------------------

    if len(sectors) == 0:
        sectors = os.listdir(mg.datapath+'clip/') # >> sigma clipped light
                                                  # >> curves
        sectors.sort()

    ticid = []
    fnames = []
    for sector in sectors:
        fnames.extend([mg.datapath+'clip/'+sector+'/'+f for f in\
                       os.listdir(mg.datapath+'clip/'+sector+'/')])
        ticid.extend([int(f[:-5]) for f in \
                      os.listdir(mg.datapath+'clip/'+sector+'/')])
    fnames.sort()

    # -- set up output directories ---------------------------------------------

    dt.create_dir(mg.savepath+'timescale-1sector/')
    savepath = mg.savepath+'timescale-1sector/lspm/'
    dt.create_dir(savepath)
    dt.create_dir(savepath+'preprocess/')
    dt.create_dir(savepath+'avg/')
    dt.create_dir(mg.datapath+'timescale-1sector/')
    lspmpath = mg.datapath+'timescale-1sector/lspm/'
    dt.create_dir(lspmpath)

    # -- determine suitable frequency grid -------------------------------------
    
    f0_avg = [] # >> average sampling rates
    T = [] # >> baselines
    for lcfile in fnames[::5000]:
        data, meta = dt.open_fits(fname=lcfile)
        time = data['TIME']
        inds = np.nonzero(~np.isnan(time))
        f0_avg.append(1/np.mean(np.diff(time[inds]))) # >> avg sampling rate
        T.append(time[inds][-1]-time[inds][0]) # >> baseline

    f_ny = 0.5*np.mean(f0_avg) # >> average nyquist frequency : 1/2 of average
                               # >> sampling rate
    T_avg = np.mean(T) # >> average baseline

    # >> frequency grid:
    min_freq = 2/T_avg
    max_freq = f_ny/2
    df = 1/(n0*T_avg)
    with open(savepath+'frequency_grid.txt', 'w') as f:
        f.write('min_freq,'+str(min_freq)+'\n')
        f.write('max_freq,'+str(max_freq)+'\n')
        f.write('df,'+str(df)+'\n')

    # -- find TICIDs observed in multiple sectors ------------------------------

    uniq_ticid, counts = np.unique(ticid, return_counts=True)
    mult_obs = uniq_ticid[np.nonzero(counts>1)]    

    # -- compute LS periodograms -----------------------------------------------
    for i in range(len(fnames)):

        if report_time and i % plot_int == 0:
            from datetime import datetime
            start = datetime.now()

        lcfile = fnames[i]

        ticid_lc = int(lcfile.split('/')[-1][:-5])
        # lcfiles = [lcfile]
        # if ticid_lc in mult_obs:
        #     lcfiles = [f for f in fnames if str(ticid_lc) in f]

        if i % plot_int == 0:
            print('Computing LS periodogram of light curve '+str(i)+'/'+\
                  str(len(fnames)))
            if plot: plot_pgram = True
            verbose = True
            gc.collect()
        else:
            plot_pgram = False
            verbose = False

        if overwrite and os.path.exists(lspmpath+str(ticid_lc)+'.fits'):
            pass
        else:
            calc_ls_pgram(mg, ticid_lc, plot=plot_pgram, verbose=verbose,
                          min_freq=min_freq, max_freq=max_freq, df=df, 
                          timescale=1, savepath=savepath, lspmpath=lspmpath)


        if report_time and i % plot_int == 0:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            print('Time to produce LS-periodogram: '+str(dur_sec))



def calc_lspgram_mult_sector(mg, plot=True, plot_int=200, sectors=[],
                             n0=6, timescale=6, overwrite=False,
                             report_time=True):

    if len(sectors) == 0:
        sectors = os.listdir(mg.datapath+'clip/') # >> sigma clipped light
                                                  # >> curves
        sectors.sort()

    # -- find TICIDs that were observed in 6 or more sectors -------------------

    ticid = []
    fnames = []
    for sector in sectors:
        fnames.extend([mg.datapath+'clip/'+sector+'/'+f for f in\
                       os.listdir(mg.datapath+'clip/'+sector+'/')])
        ticid.extend([int(f[:-5]) for f in \
                      os.listdir(mg.datapath+'clip/'+sector+'/')])
    fnames.sort()

    unq_ticid, counts = np.unique(ticid, return_counts=True)
    inds = np.nonzero(counts >= timescale)
    ticid = np.array(unq_ticid)[inds]

    print('Number of TICIDs: '+str(len(ticid)))

    # -- set up output directories ---------------------------------------------
    dt.create_dir(mg.savepath+'timescale-'+str(timescale)+'sector/')
    savepath = mg.savepath+'timescale-'+str(timescale)+'sector/lspm/'
    dt.create_dir(savepath)
    dt.create_dir(savepath+'preprocess/')
    # dt.create_dir(savepath+'concat/')
    dt.create_dir(mg.datapath+'timescale-'+str(timescale)+'sector/')
    lspmpath = mg.datapath+'timescale-'+str(timescale)+'sector/lspm/'
    dt.create_dir(lspmpath)

    # -- determine suitable frequency grid -------------------------------------

    f0_avg = [] # >> average sampling rates
    T = [] # >> baselines

    for ticid_lc in ticid[::200]:
        lcfiles = [f for f in fnames if str(ticid_lc) in f][:timescale]
        time = []
        for lcfile in lcfiles: 
            data, meta = dt.open_fits(fname=lcfile)
            time.append(data['TIME'])
        time = np.concatenate(time)
        inds = np.nonzero(~np.isnan(time))
        f0_avg.append(1/np.mean(np.diff(time[inds]))) # >> avg sampling rate
        T.append(time[inds][-1]-time[inds][0]) # >> baseline

    f_ny = 0.5*np.mean(f0_avg) # >> average nyquist frequency : 1/2 of average
                               # >> sampling rate
    T_avg = np.mean(T) # >> average baseline

    # >> frequency grid:
    min_freq = 2/T_avg
    max_freq = f_ny/2
    df = 1/(n0*T_avg)
    with open(savepath+'frequency_grid.txt', 'w') as f:
        f.write('min_freq,'+str(min_freq)+'\n')
        f.write('max_freq,'+str(max_freq)+'\n')
        f.write('df,'+str(df)+'\n')

    print('Min period: '+str(np.round(1/max_freq * 1440)) + ' minutes')
    print('Max period: '+str(np.round(1/min_freq)) + ' days')

    # -- compute LS periodograms -----------------------------------------------
    for i in range(len(ticid)):

        if report_time and i % plot_int == 0:
            from datetime import datetime
            start = datetime.now()

        ticid_lc = int(ticid[i])
        # lcfiles = [f for f in fnames if str(ticid_lc) in f][:timescale]

        if i % plot_int == 0:
            print('Computing LS periodogram of light curve '+str(i)+'/'+\
                  str(len(ticid)))
            if plot: plot_pgram = True
            verbose = True
        else:
            plot_pgram = False
            verbose = False

        # !! tmp
        if overwrite and os.path.exists(lspmpath+str(ticid_lc)+'.fits') \
           and not plot_pgram:
            pass
        else:
            calc_ls_pgram(mg, ticid_lc, plot=plot_pgram, 
                             min_freq=min_freq, max_freq=max_freq, df=df, 
                             timescale=timescale, savepath=savepath, 
                             lspmpath=lspmpath, verbose=verbose)

        if report_time and i % plot_int == 0:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            print('Time to produce LS-periodogram: '+str(dur_sec))

def calc_ls_pgram(mg, ticid, plot=False, meta=None,
                  max_freq=1/(8/1440.), min_freq=1/12.7,
                  df=1/(4*27.), savepath=None, lspmpath=None,
                  verbose=True, timescale=1):
    '''
    * min_freq : default it 8 minutes (~> average Nyquist frequency)
    * max_freq : default is 27 days (~ average baseline)
    * df : default
    '''

    # >> open light curve files
    time, flux = get_lc(mg.datapath+'clip/', ticid, timescale, rmv_nan=True,
                        detrend=True, plot=plot, savepath=savepath,
                        return_sector=True, norm=True, method='standardize')

    time, flux = np.array(time), np.array(flux)

    freq = np.arange(min_freq, max_freq, df)

    if timescale == 1:
        lspm_sector = []
        for i in range(len(time)):
            num_inds = np.nonzero(~np.isnan(flux[i]))
            lspm_sector.append(LombScargle(time[i][num_inds],
                                           flux[i][num_inds]).power(freq))
        lspm = np.median(lspm_sector, axis=0)
    else:
        num_inds = np.nonzero(~np.isnan(flux))
        lspm = LombScargle(time[num_inds],
                           flux[num_inds]).power(freq)

    # >> save periodogram
    dt.write_fits(lspmpath, meta, [freq, lspm], ['FREQ', 'LSPM'],
                  verbose=verbose, fname=str(ticid)+'.fits')

    gc.collect()

    if plot: # -- preprocessing plot -------------------------------------------
        fig, ax = plt.subplots(3)

        time_mask, flux_mask = get_lc(mg.datapath+'mask/', ticid, timescale,
                                      rmv_nan=True)
        ax[0].set_title('Raw light curve (with quality flag mask)')
        plot_lc(ax[0], time_mask, flux_mask)

        ax[1].set_title('Sigma clipped and detrended light curve')
        if timescale == 1:
            plot_lc(ax[1], np.concatenate(time), np.concatenate(flux))
        else:
            plot_lc(ax[1], time, flux)

        ax[2].set_title('LS-periodogram')
        plot_lspm(ax[2], freq, lspm)
        fig.tight_layout()
        fname = savepath+'preprocess/preprocess_TIC'+str(int(ticid))+'.png'
        fig.savefig(fname)
        print('Saved '+fname)
        plt.close(fig)

        lcfiles = []
        for s in os.listdir(mg.datapath+'clip/'): # >> loop through sectors
            lcfiles.extend([mg.datapath+'clip/'+s+'/'+f for f in \
                           os.listdir(mg.datapath+'clip/'+s) \
                           if str(ticid) in f])

    if plot and timescale == 1 and len(lcfiles)>1 : # -- averaging LS-pm ---
        sector_tic = [int(f.split('/')[-2].split('-')[-1]) for f in lcfiles]
        nrows = len(time)+1
        fig, ax = plt.subplots(nrows, 2, figsize =(9*2,4*nrows))
        for i in range(len(time)):
            ax[i,0].set_title('TIC '+str(ticid)+' LS-periodogram, Sector '+\
                            str(sector_tic[i]))
            plot_lspm(ax[i,0], freq, lspm_sector[i])
            ax[i,1].set_title('TIC '+str(ticid)+' light curve, Sector '+\
                            str(sector_tic[i]))
            plot_lc(ax[i,1], time[i], flux[i])

        ax[-1,0].set_title('TIC '+str(ticid)+' LS-periodogram, averaged')
        plot_lspm(ax[-1,0], freq, lspm)
        ax[-1,1].axis('off')
        fig.tight_layout()
        fname = savepath+'avg/lspgram_avg_TIC'+str(ticid)+'.png'
        fig.savefig(fname)
        print('Saved '+fname)
        plt.close(fig)

    # if plot and timescale > 1:
    #     sector_tic = [int(f.split('/')[-2].split('-')[-1]) for f in lcfiles]
    #     nrows = len(time)+1
    #     fig, ax = plt.subplots(nrows, figsize=(9,4*nrows))
    #     for i in range(len(time)):
    #         ax[i].set_title('TIC '+str(ticid)+' light curve, Sector '+\
    #                         str(sector_tic[i]))
    #         plot_lc(ax[i], time[i], flux[i])

    #     ax[-1].set_title('TIC '+str(ticid)+' LS-periodogram\nSectors '+\
    #                      '_'.join(np.array(sector_tic).astype('str')))
    #     plot_lspm(ax[-1], freq, lspm)
    #     fig.tight_layout()
    #     fname = savepath+'concat/lspgram_concat_TIC'+str(ticid)+'.png'
    #     fig.savefig(fname)
    #     print('Saved '+fname)
    #     plt.close(fig)

def detrend_lc(time, flux):
    from scipy.signal import detrend

    # >> find gaps of a day
    inds = np.nonzero(np.diff(time) > 1.)[0]
    for i in range(len(inds)-1):
        if i == 0:
            gap_inds = np.arange(0, inds[i]+1, dtype='int')
        else:
            gap_inds = np.arange(inds[i-1], inds[i]+1, dtype='int')
        flux[gap_inds] = detrend(flux[gap_inds])

    return flux

def local_rms(x, y, kernel):
    local_rms = []
    for i in range(len(x)):
        y = detrend_lc(x, y)
        local_rms.append(np.sqrt(np.mean(y[np.max([0,i-kernel]):\
                                           np.min([len(x),i+kernel])]**2)))
    rms = np.mean(local_rms)
    return rms

def local_std(x, y, kernel):
    local_std = []
    for i in range(len(x)):
        local_std.append(np.std(y[np.max([0,i-kernel]):\
                                  np.min([len(x),i+kernel])]))
    std = np.mean(local_std)
    return std

def preprocess_lspgram(mg, n_chunk=10, plot_int=1000,
                       timescale=1):

    lspmpath = mg.datapath+'timescale-'+str(timescale)+'sector/lspm/'
    datapath = mg.datapath+'timescale-'+str(timescale)+'sector/ae/'
    dt.create_dir(datapath)

    fnames = [lspmpath+f for f in os.listdir(lspmpath)]
    ticid = [f[:-5] for f in os.listdir(lspmpath)]

    savepath = mg.savepath+'timescale-'+str(timescale)+'sector/normalize/'
    dt.create_dir(savepath)

    # >> save LS periodograms in chunks
    for n in range(n_chunk): 
        if n == n_chunk-1:
            fnames_chunk = fnames[n*(len(fnames)//n_chunk):]
        else:
            fnames_chunk = fnames[n*(len(fnames)//n_chunk):\
                                  (n+1)*(len(fnames)//n_chunk)]
        lspgram, sector, ticid = [], [], []
        for i in range(len(fnames_chunk)):
            with fits.open(fnames_chunk[i], memmap=False) as hdul:
                freq = hdul[1].data['FREQ']
                lspgram.append(hdul[1].data['LSPM'].astype(np.float32))
                # sector.append(hdul[0].header['SECTOR'])
                tic = int(fnames_chunk[i].split('/')[-1].split('.')[0])
                ticid.append(tic)

            if i % plot_int == 0:
                print('Chunk '+str(n)+' / '+str(n_chunk-1)+': '+\
                      str(i)+' / '+str(len(fnames_chunk)))
                
                gc.collect() # >> prevents slowing

                fig, ax = plt.subplots(2)
                ax[0].set_title('Unnormalized LS periodogram for TIC '+\
                                str(tic))
                plot_lspm(ax[0], freq, hdul[1].data['LSPM'])

                ax[1].set_title('Normalized LS periodogram for TIC '+\
                                str(tic))
                # plot_lspm(ax[1], freq,  dt.standardize([hdul[1].data['LSPM']])[0])
                plot_lspm(ax[1], freq,  dt.normalize_minmax([hdul[1].data['LSPM']],
                                                            new_min=-1.)[0])
                fig.tight_layout()
                fig.savefig(savepath+'TIC'+str(tic)+'.png')
                print(savepath+'TIC'+str(tic)+'.png')
                plt.close(fig)

        # >> convert lists to numpy arrays
        trunc = np.min([len(l) for l in lspgram]) # !! hard code
        lspgram = [l[:trunc] for l in lspgram] # !! hard code
        lspgram  = np.array(np.stack(lspgram))
        sector, ticid = np.array(sector), np.array(ticid)

        # >> standardize # >> drives most values negative
        # lspgram = dt.standardize(lspgram, ax=1)

        # >> normalize
        lspgram = dt.normalize_minmax(lspgram, new_min=-1., new_max=1.)

        # >> save
        np.save(datapath+'chunk%02d'%n+'_train_lspm.npy', lspgram)
        np.save(datapath+'chunk%02d'%n+'_train_sector.npy', sector)
        np.save(datapath+'chunk%02d'%n+'_train_ticid.npy', ticid)
        np.save(datapath+'chunk%02d'%n+'_train_freq.npy', freq)

def load_lspgram_fnames(mg, timescale=1):
    path = mg.datapath+'timescale-'+str(timescale)+'sector/ae/'
    n_chunks = max([int(f[5:7]) for f in os.listdir(path) if 'chunk' in f])+1
    
    fnames, mg.sector, mg.objid = [], [], []
    for n in range(n_chunks):
        mg.freq = np.load(path+'chunk%02d'%n+'_train_freq.npy')
        fnames.append(path+'chunk%02d'%n+'_train_lspm.npy')
        mg.sector.extend(np.load(path+'chunk%02d'%n+'_train_sector.npy'))
        mg.objid.extend(np.load(path+'chunk%02d'%n+'_train_ticid.npy'))

    mg.batch_fnames = fnames
    mg.sector = np.array(mg.sector).astype('int')
    mg.objid = np.array(mg.objid).astype('int')
    mg.x_train = None
        
def load_lspgram(mg):
    path = mg.datapath+'dae/'
    n_chunks = max([int(f[5:7]) for f in os.listdir(path) if 'chunk' in f])+1
    
    mg.sector, mg.objid, mg.x_train = [], [], []
    for n in range(n_chunks):
        mg.freq = np.load(path+'chunk%02d'%n+'_train_freq.npy')
        mg.sector.extend(np.load(path+'chunk%02d'%n+'_train_sector.npy'))
        mg.objid.extend(np.load(path+'chunk%02d'%n+'_train_ticid.npy'))
        mg.x_train.extend(np.load(path+'chunk%02d'%n+'_train_lspm.npy'))

    mg.sector = np.array(mg.sector).astype('int')
    mg.objid = np.array(mg.objid).astype('int')
    # mg.x_train = np.array(mg.x_train).astype(np.float32)

    # mg.x_train = mg.x_train.astype(np.float32)
    # mg.x_train = np.expand_dims(mg.x_train, axis=-1)
    # !! testing 1
    # mg.freq = mg.freq[:50000]
    # mg.x_train = np.array([z[:50000] for z in mg.x_train])
    # !! testing 2
    # mg.sector = mg.sector[:100]
    # mg.objid = mg.objid[:100]
    # mg.x_train = mg.x_train[:100].astype(np.float32)

    mg.sector = mg.sector[:1000]
    mg.objid = mg.objid[:1000]
    mg.x_train = np.array(mg.x_train[:1000]).astype(np.float32)
            
# -- Phase Curve Features ------------------------------------------------------

def create_phase_curve_feats(datapath, timescale, 
                             time=None, flux=None, n_bins=2000,
                             n_freq=500000, 
                             n_terms=1, n_freq0=10000, n_terms0=1,
                             sector=1, output_dir='./', plot=False,
                             plot_int=200,
                             report_time=False):

    output_dir='timescale-'+str(timescale)+'sector/lspm-feat/'
    dt.create_dir(output_dir)

    # >> textfile showing TICIDs where phase curve features were computed
    fname = output_dir+'phase_curve_feat_gen.txt'
    with open(fname, 'w') as f: 
        f.write('')

    lspmpath = datapath+'timescale-'+str(timescale)+'sector/lspm/'
    ticid = [int(f[:-5]) for f in os.listdir(lspmpath)]

    period_ticid = []
    phase_curves = []
    feats = []
    # >> calculate phase curves
    for i in range(len(ticid)):

        if i % plot_int == 0:
            plot, report_time = True, True
        else: 
            plot, report_time = False, False

        if report_time: # >> restart timer
            start = datetime.now()

        # >> load LS periodogram 
        hdul = fits.open(lspmpath+str(ticid[i])+'.fits')
        frequency = hdul[1].data['FREQ']
        power = hdul[1].data['LSPM']

        calc_lspm_stats(frequency=frequency, power=power, datapath=datapath,
                     ticid=ticid[i], output_dir=output_dir)

        # # >> check if light curve is periodic
        # res = mask_harmonics(frequency=frequency, power=power, ticid=ids[i],
        #                      report_time=report_time, plot=plot,
        #                      output_dir=output_dir, datapath=datapath)

        # # >> calculate phase curve 
        # feats = calc_phase_curve(frequency=frequency, power=power, ticid=ids[i],
        #                          report_time=report_time, plot=plot,
        #                          output_dir=output_dir, datapath=datapath)

        # with open(fname, 'a') as f:
        #     f.write(str(ids[i])+','+str(type(feats)==type(None)))
        # if not type(feats) == type(None):
        #     all_feats.append(feats)

        if report_time:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            print('Time to make phase curve: '+str(dur_sec))


    # feats = dt.standardize(feats, ax=0)
    # all_feats = dt.standardize(all_feats, ax=0)
    # np.savetxt(output_dir+'Sector'+str(sector)+'_phase_curve_feat.txt',
    #            np.array(all_feats))

def calc_lspm_stats(t=None, y=None, frequency=None, power=None,
                       datapath=None, ticid=None,
                       timescale=1, n_freq=10000, n_terms=1,
                       thresh_min=2, thresh_max=5e-2, thresh_std=55,
                       tmin=0.05, tmax=2, har_window=3, kernel_size=5,
                    max_window=20, thresh_rms=20,
                       report_time=True, plot=True, output_dir='',
                       plot_freq=False):
    ''' Iteratively mask harmonics of the largest power to determine whether it
    is appropriate to compute phase curve features. Should take ~1 ms for short-
    cadence light curves.


    thresh_min : minimum threshold power for a peak (used to determine the width
                 of maximum peak)
    thresh_max : maximum threshold power for a peak (used to determine if there is
                 more than one signficiant component in periodogram'''

    from astropy.timeseries import LombScargle
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    from scipy.signal import medfilt
    from scipy.signal import detrend
    from scipy.stats import skew, kurtosis
    from scipy.optimize import curve_fit
    import astropy.units as u
    import itertools as it
    
    if report_time: # >> start timer
        from datetime import datetime
        start = datetime.now()

    # -- calculate a sparse periodogram ----------------------------------------
    if type(frequency) == type(None) and type(t) == type(None):
        hdul = fits.open(datapath+'timescale-'+str(timescale)+'sector/lspm/'+\
                         str(ticid)+'.fits')
        frequency = hdul[1].data['FREQ']
        power = hdul[1].data['LSPM']
    elif type(frequency) == type(None):
        frequency = np.linspace(1./tmax, 1./tmin, n_freq)
        power = LombScargle(t, y, nterms=n_terms).power(frequency)    
    
    # -- find peaks ------------------------------------------------------------
    periodic = True # >> assume periodicity
    dtrn_pow = []
    masked_pow = []
    smooth_pow = []
    peaks_freq = []
    peaks_ind = []
    peaks_amp = []
    peaks_max_wind = []
    peaks_har_wind = []
    peaks_har_freq = []
    masked_power = np.copy(power)

    # >> values for checking periodicity
    fwhm = np.min(frequency)/2 # >> (1/T)/2 = min_freq/2
    df = frequency[1]-frequency[0]
    kernel = int(5*fwhm/df) # ~ 5*

    detrend_power = detrend(masked_power) 
    rms = local_rms(frequency, detrend_power, kernel)

    while periodic:
        # >> find frequency with the highest power
        max_pow = np.max(masked_power)
        max_ind = np.argmax(masked_power)
        max_freq = frequency[max_ind] 

        # -- check if there is a peak at max_freq (check for periodicity) ------
        detrend_power = detrend(masked_power) 

        # if len(peaks_freq) == 1: pdb.set_trace()
        # if (np.max(detrend_power)-np.median(detrend_power)) < \
        #    thresh_std*np.std(detrend_power) or\
        #    len(peaks_freq) >= 10: 
        if detrend_power[max_ind] < thresh_rms*rms or len(peaks_freq) >= 10:
            periodic = False
        else: # >> mask max peak and its harmonics, then find next peak
            peaks_freq.append(max_freq)
            peaks_ind.append(max_ind)
            peaks_amp.append(masked_power[max_ind])
            dtrn_pow.append(detrend_power)

            # -- mask harmonics ------------------------------------------------
            factors = [2, 3]
            df = np.diff(frequency)[0]
            windows = []
            peak_f = []
            peak_widths =[]
            peak_har = []
            har_freq = []
            for factor in factors:
                har_ind = np.argmin(np.abs(frequency - max_freq/factor))
                window = np.arange(np.max([0,har_ind-har_window]),
                                   np.min([har_ind+har_window, len(frequency)]),
                                   dtype='int')
                
                windows.append(window)
                peak_har.append(window)
                
                har_freq.append(har_ind)


                har_ind = np.argmin(np.abs(frequency - max_freq*factor))

                window = np.arange(np.max([0,har_ind-har_window]),
                                   np.min([har_ind+har_window, len(frequency)]),
                                   dtype='int')
                windows.append(window)
                peak_har.append(window)
                har_freq.append(har_ind)
            peaks_har_wind.append(peak_har)
            peaks_har_freq.append(har_freq)

            # >> frequency inds corresponding to harmonics
            inds = np.array([i for wind in windows for i in wind]).astype('int')
            inds = np.unique(inds)

            # -- mask maximum peak ---------------------------------------------
            # >> find width of maximum peak
            smoothed_power = medfilt(masked_power, kernel_size=3)
            smooth_pow.append(smoothed_power)
            # smoothed_power = masked_power
            # no_peak_inds = np.nonzero((smoothed_power-\
            #                            thresh_min*np.max(smoothed_power))<0)[0]
            no_peak_inds = np.nonzero(smoothed_power<thresh_min*rms)[0]
            sorted_inds = no_peak_inds[np.argsort(np.abs(no_peak_inds-max_ind))]
            if max_ind < np.min(sorted_inds):
                left_ind = 0
            else:
                left_ind = sorted_inds[np.nonzero(sorted_inds < max_ind)[0][0]]
            if max_ind > np.max(sorted_inds):
                right_ind = len(masked_power)-1
            else:
                 right_ind = sorted_inds[np.nonzero(sorted_inds>max_ind)[0][0]]
            window = np.arange(left_ind, right_ind)
            windows.append(window)
            inds = np.append(inds, window)

            peaks_max_wind.append(window)

            # -- mask out maximum peak and harmonics ---------------------------

            # masked_power[inds] = np.min(masked_power)
            masked_power[inds] = rms

            masked_pow.append(np.copy(masked_power))

    # -- compute LS-periodogram statistics  -------------------------------------

    num_peaks = len(peaks_freq)
    peaks_period = 1/np.array(peaks_freq)
    peaks_amp = np.array(peaks_amp)
    lspm_stats = [num_peaks]
    
    if len(peaks_period) == 0: # >> not periodic
        lspm_stats.extend([np.nan]*16)
    else:
        if len(peaks_period) >= 2:
            peaks_p_comb = pd.Series(list(it.combinations(peaks_period,2))).values
            peaks_period_ratios = [p[0]/p[1] for p in peaks_p_comb]

            peaks_a_comb = pd.Series(list(it.combinations(peaks_amp,2))).values
            peaks_amp_ratios = [a[0]/a[1] for a in peaks_a_comb]
        else:
            peaks_period_ratios = [np.nan]*4
            peaks_amp_ratios = [np.nan]*4
        for feat in [peaks_period, peaks_period_ratios, peaks_amp, 
                      peaks_amp_ratios]:
            lspm_stats.append(np.mean(feat))
            lspm_stats.append(np.std(feat))
            lspm_stats.append(skew(feat))
            lspm_stats.append(kurtosis(feat))   

    # -- plot ------------------------------------------------------------------
    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        print('Time to find peaks: '+str(dur_sec))

    if plot:
        fig, ax = plt.subplots(len(peaks_freq)+1, 3,
                               figsize=(24, 5*(len(peaks_freq)+1)))

        if type(t) == type(None):
            t, y = get_lc(datapath+'clip/', ticid, timescale, norm=True,
                          rmv_nan=True, method='standardize')

        if len(peaks_freq) == 0:
            ax = np.array([ax])

        ax[0, 0].set_title('Light curve for TIC '+str(ticid))
        plot_lc(ax[0, 0], t, y)
        ax[0, 1].axis('off')
        ax[0, 2].set_title('LS-periodogram for TIC '+str(ticid))
        plot_lspm(ax[0, 2], frequency, power, plot_freq=plot_freq)


        for i in range(len(peaks_freq)):
            # >> phase curve
            period = 1/peaks_freq[i]
            folded_t, folded_y = calc_phase_curve(period, time=t, flux=y,
                                                  freq=frequency, lspm=power)

            plot_lc(ax[i+1, 0], folded_t, folded_y, c='k', marker='.', ms=2,

                    fillstyle='full', label='data', linestyle='')

            w = 2*np.pi/period
            # def sinfunc(t, A, w, p, c): return A*np.sin(w*t + p) + c
            # popt, pcov = curve_fit(sinfunc, folded_t, folded_y,
            #                        p0=[1,1,2*np.pi/period,1])
            # A, w, p, c = popt
            # sinfit = sinfunc(folded_t, A, w, p, c)

            def sinfunc(t, A, p, c): return A*np.sin(w*t + p) + c
            popt, pcov = curve_fit(sinfunc, folded_t, folded_y)

            A, p, c = popt
            sinfit = sinfunc(folded_t, A, p, c)
            plot_lc(ax[i+1, 0], folded_t, sinfit, c='r', linestyle='-',
                    label='fit')
            ax[i+1, 0].set_xlabel('Time from midpoint epoch [days]')
            ax[i+1, 0].legend()
            chi2 = np.abs(np.sum((sinfit-folded_y)**2\
                                 /folded_y))
            ax[i+1, 0].set_title('Phase curve with period '+\
                                 str(np.round(period,3))+\
                                 ' days\nChi-squared: '+str(np.round(chi2, 3)))

            # >> intermediate plot
            plot_lspm(ax[i+1, 1], frequency, dtrn_pow[i], plot_freq=plot_freq,
                      c='k', label='detrended')
            plot_lspm(ax[i+1, 1], frequency, masked_pow[i], plot_freq=plot_freq,
                      c='r', alpha=0.5, label='masked')
            plot_lspm(ax[i+1, 1], frequency, smooth_pow[i], plot_freq=plot_freq,
                      c='b', label='smooth')
            ax[i+1, 1].axhline(rms, linestyle='--', label='RMS')
            ax[i+1, 1].axhline(thresh_rms*rms, linestyle='--',
                               label=str(thresh_rms)+'*RMS')

            ax[i+1, 1].legend(prop={'size':'xx-small'})

            # >> ls periodogram
            plot_lspm(ax[i+1, 2], frequency, power, plot_freq=plot_freq)

            windows = [peaks_max_wind[i]]
            windows.extend(peaks_har_wind[i])


            har_periods = []
            for j in range(len(windows)):
                window = windows[j]
                if j == 0:
                    ind = peaks_ind[i]
                    freq = peaks_freq[i]
                    c='r'
                else:
                    ind = peaks_har_freq[i][j-1]
                    freq = frequency[ind]
                    c='b'
                if plot_freq:
                    ax[i+1, 2].axvspan(frequency[window[0]],
                                       frequency[window[-1]],
                                       alpha=0.2, facecolor=c)
                else:
                    ax[i+1, 2].axvspan(1/frequency[window[-1]],
                                       1/frequency[window[0]],
                                       alpha=0.2, facecolor=c)

                if j != 0:
                    har_periods.append(str(np.round(1/freq, 3)))
                # yloc = np.min(power)+0.5*(power[ind] - np.min(power))
                # yloc = np.min(power)
                yloc = np.exp((np.log(np.min(power)) - np.log(power[ind]))/2)
                if plot_freq:
                    ax[i+1, 2].axvline(freq, alpha=0.4, c=c, linestyle='dashed')
                    ax[i+1, 2].text(freq, yloc, str(np.round(1/freq,2))+\
                                    '\ndays', fontsize='xx-small', ha='center',
                                    va='center')
                else:
                    ax[i+1, 2].axvline(1/freq, alpha=0.4, c=c)
                    ax[i+1, 2].text(1/freq, yloc, str(np.round(1/freq,3))+\
                                    '\ndays', fontsize='small')


            har_periods = ', '.join(har_periods)
            ax[i+1, 2].set_title('Max-power peak (red) at\n'+\
                                 str(np.round(period,3))+' days\n'+\
                                 'Harmonics (blue) at\n'+har_periods+\
                                 ' days')

        
        fig.tight_layout()
        out_f = output_dir+'harmonics_TIC'+str(ticid)+'.png'
        fig.savefig(out_f)
        print('Saved '+out_f)
        plt.close(fig)


    return peaks_period, lspm_stats
    
    # if np.max(np.delete(power, inds)) > max_pow*thresh_max:
    #     periodic = False

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

# def calc_phase_curve_feats(t=None, y=None, frequency=None, power=None,
#                      ticid=None, timescale=1,
#                      n_bins=100, n_freq=50000, n_terms=2,
#                      plot=False, output_dir='', report_time=True,
#                      tmin=0.05, tmax=27):
#     '''frac_max : adjusts the threshold power to determine the width of maximum peak
#     frac : adjusts the threshold power to determine the signfiicance of second
#     largest peak
    
#     Phase curve parameters from Chen et al., 2020
#     https://doi.org/10.3847/1538-4365/ab9cae
#     * period (logP)
#     * amplitude ratio (R_21 = a2/a1)
#     * amplitude (Amp.)
#     * magnitude (mag)
#     * Adjusted R^2

#     '''

#     # >> temporal baseline is 27 days, and shortest timescale sensitive to is
#     # >> 4 minutes
#     #tmax = 27 # days
#     # tmin = 4./1440 # days
#     #tmin = 0.025

#     from astropy.timeseries import LombScargle
#     from astropy.timeseries import TimeSeries
#     from astropy.time import Time
#     import astropy.units as u

#     if report_time: # >> restart timer
#         start = datetime.now()

#     # -- compute periodogram ---------------------------------------------------
#     if type(frequency) == type(None):
#         num_indx = np.nonzero(~np.isnan(y))
#         t, y = t[num_indx], y[num_indx]
#         frequency, power = LombScargle(t, y, nterms=n_terms).autopower()

#     peak_c = np.argmax(power)
#     peak_freq = np.linspace(frequency[peak_c-100], frequency[peak_c+100], 100000)
#     pow_peak = LombScargle(t,y,nterms=n_terms).power(peak_freq)
#     best_freq = peak_freq[np.argmax(pow_peak)]
    
#     period = 1/best_freq

#     # -- compute phase curve ---------------------------------------------------

#     # >> bin phase curve
#     orig_len = ts_folded['flux'].shape[0]
#     new_len = orig_len - orig_len%n_bins

#     folded = ts_folded['flux'][np.argsort(ts_folded.time.value)]
#     pc_feats = np.array(np.split(folded[:new_len], n_bins))
#     pc_feats = np.mean(pc_feats, axis=1)

#     if plot:
#         fig, ax = plt.subplots(4, figsize=(8, 4*3))
#         if report_time:
#             ax[0].set_title('Computation time: '+str(dur_sec)) 
#         ax[0].plot(t, y, '.k', ms=1)
#         ax[0].set_xlabel('Time [BJD - 2457000]')
#         ax[0].set_ylabel('Relative Flux')
#         ax[1].set_title('Period: '+str(np.round(period, 3))+' days')
#         ax[1].plot(frequency, power, '.k', ms=1)
#         ax[1].set_xlabel('Frequency')
#         ax[1].set_ylabel('Power')
#         # ax[1].set_xscale('log')
#         ax[2].plot(ts_folded.time.value, ts_folded['flux'], '.k', ms=1)
#         ax[2].set_xlabel('Time from midpoint epoch [days]')
#         ax[2].set_ylabel('Relative Flux')
#         ax[3].plot(np.arange(len(pc_feats)), pc_feats, '.')
#         ax[3].set_ylabel('Binned phase curve (nbins=' +str(n_bins)+ ')')
#         fig.tight_layout()
#         out_f = output_dir+'phase_curve_TIC'+str(int(ticid))+'.png'
#         fig.savefig(out_f)
#         print('Saved '+out_f)
#         plt.close(fig)

#     return pc_feats

def calc_phase_curve(period, time=None, flux=None, freq=None, lspm=None,
                     datapath=None, ticid=None, timescale=1, report_time=True):
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    from datetime import datetime 

    if type(time) == type(None):
        time, flux = get_lc(datapath+'clip/', ticid, timescale,
                            norm='standardize', rmv_nan=True)

    if report_time:
        start = datetime.now()

    time = Time(time, format='jd') 
    ts = TimeSeries(time=time, data={'flux': flux})
    ts_folded = ts.fold(period=period*u.d) 

    # sorted_inds = np.argsort(ts_folded.time.value)
    sorted_inds = np.argsort(ts_folded['time'])
    folded_t = ts_folded['time'][sorted_inds]

    # folded = ts_folded['flux'][np.argsort(ts_folded.time.value)]
    folded_y = ts_folded['flux'][sorted_inds]

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        print('Time to make phase curve: '+str(dur_sec))

    return folded_t.value, folded_y.value

def test_simulated_data(savepath, datapath, timescale=1):
    
    # >> get frequency grid
    with open(savepath+'timescale-'+str(timescale)+'sector/lspm/'+\
              'frequency_grid.txt', 'r') as f:
        lines = f.readlines()
        min_freq = float(lines[0].split(',')[1][:-2])
        max_freq = float(lines[1].split(',')[1][:-2])
        df = float(lines[2].split(',')[1][:-2])
    freq = np.arange(min_freq, max_freq, df)

    # >> get time grid
    T = 2/min_freq
    sampling_rate = 1/(4*max_freq)
    t = np.arange(0, T, sampling_rate)

    # >> create output directory
    dt.create_dir(savepath+'timescale-'+str(timescale)+'sector/feat_eng/')
    savepath = savepath+'timescale-'+str(timescale)+'sector/feat_eng/'+\
               'lspm-feats-mock/'
    dt.create_dir(savepath)

    for i in range(4):
        if i == 0: # >> random noise
            y = np.random.normal(size=len(t))
            ticid = '001_random_normal'
        elif i == 1: # >> period = 3 days
            y = np.random.normal(size=len(t)) + np.sin( ((2*np.pi)/3) * t)
            ticid = '002_period_3d'
        elif i == 2: # >> periods = 3, 7 days
            y = np.random.normal(size=len(t)) + np.sin( ((2*np.pi)/3) * t) + \
                np.sin( ((2*np.pi)/7) * t)
            ticid = '003_period_3d_7d'
        elif i == 3: # >> periods = 2, 7 days
            y = 0.5*np.random.normal(size=len(t)) + np.sin( ((2*np.pi)/2) * t) + \
                0.7*np.sin( ((2*np.pi)/7) * t)
            ticid = '004_period_2d_7d'
        y = dt.standardize(y, ax=0)
        power = LombScargle(t, y).power(freq)
        calc_lspm_stats(t=t, y=y, frequency=freq, power=power, datapath=datapath,
                     ticid=ticid, output_dir=savepath)

    return

def spoc_noise(datapath, kernel=30):
    from scipy.signal import medfilt

    lcpath = datapath+'clip/'

    fig, ax = plt.subplots()


def periodic_simulated_data(savepath, datapath, metapath, timescale=1,
                            n_samples=100):
    from scipy.stats import loguniform
    
    # >> get frequency grid
    with open(savepath+'timescale-'+str(timescale)+'sector/lspm/'+\
              'frequency_grid.txt', 'r') as f:
        lines = f.readlines()
        min_freq = float(lines[0].split(',')[1][:-2])
        max_freq = float(lines[1].split(',')[1][:-2])
        df = float(lines[2].split(',')[1][:-2])
    freq = np.arange(min_freq, max_freq, df)

    # >> get time grid
    T = 2/min_freq
    sampling_rate = 1/(4*max_freq)
    t = np.arange(0, T, sampling_rate)

    # >> create output directory
    dt.create_dir(savepath+'timescale-'+str(timescale)+'sector/feat_eng/')
    savepath = savepath+'timescale-'+str(timescale)+'sector/feat_eng/'+\
               'lspm-feats-mock/'
    dt.create_dir(savepath)

    # -- infer white noise amplitude -------------------------------------------

    # >> infer typical noise levels from Sector 1 light curves' local stdev 
    sector = 1

    if os.path.exists(savepath+'Sector%02d'%sector+'_Tmag.txt'):
        filo = np.loadtxt(savepath+'Sector%02d'%sector+'_Tmag.txt',
                          delimiter=',', skiprows=1)
        mags = filo[:,0]
        white_noise_amp = filo[:,1]
        mag_min, mag_max, mag_bin = min(mags), max(mags), np.diff(mags)[0]
        
    else:
        mag_min, mag_max = 2.5, 17.5 # >> Sector 1 Tmag: [1.94, 18.16]
        mag_bin = 0.5 # >> mags sampled from min=mag_min, max=mag_max, step=mag_bin
        
        lcpath = datapath+'clip/'
        filo = np.loadtxt(metapath+'spoc/targ/2m/all_targets_S%03d'%sector+\
                          '_v1.txt')
        ticid = filo[:,0].astype('int')
        Tmag = filo[:,3] # >> TESS magnitudes

        ticid_lc = [int(f[:-5]) for f in os.listdir(lcpath+'sector-%02d'%sector+'/')]
        _, comm1, comm2 = np.intersect1d(ticid, ticid_lc, return_indices=True)
        ticid = ticid[comm1]
        Tmag = Tmag[comm1]

        inds = np.argsort(Tmag) # >> sort ticid and Tmag by Tmag
        ticid = ticid[inds]
        Tmag = Tmag[inds]
        grid = np.indices(ticid.shape)[0]

        mags = np.arange(mag_min, mag_max, mag_bin)
        white_noise_amp = []
        fig, ax = plt.subplots(len(mags), 3, figsize=(len(mags)*4, 3*10))
        for i, mag in zip(range(len(mags)), mags):
            inds = np.nonzero((Tmag < mag+mag_bin)*(Tmag > mag-mag_bin))[0]
            std_avg = []
            for j in range(len(inds)):
                t, y = get_lc(lcpath, ticid[inds[j]], 1, return_sector=True,
                                 rmv_nan=True)        
                t, y = t[0], y[0]

                std_lc = local_std(t, y, 15)
                std_avg.append(std_lc)
                if j < 3:
                    plot_lc(ax[i,j], t, y)
                    ax[i,j].set_title('TIC '+str(ticid[inds[j]])+\
                                      '\nTmag '+str(Tmag[inds[j]])+\
                                      ', Local STD '+str(np.round(std_lc, 3)))

            std_avg = np.mean(std_avg)
            ax[i,0].set_title('White noise amplitude: '+str(std_avg)+'\n'+\
                              ax[i,0].get_title())
            white_noise_amp.append(std_avg)
        fig.tight_layout()
        fig.savefig(savepath+'Sector%02d'%sector+'_Tmag_lc.png')
        np.savetxt(savepath+'Sector%02d'%sector+'_Tmag.txt',
                   np.array([mags, white_noise_amp]).T, delimiter=',',
                   header='Tmag,WN_amp')

    # -- Initialize distributations --------------------------------------------
    # >> Initilize log uniform random variables to sample number of peaks,
    # >> amplitudes, periods, and magnitudes
    rv_npeak = loguniform.rvs(0, 10, size=n_samples)
    rv_mag = loguniform.rvs(mag_min, mag_max, size=n_samples)
    rv_mag = (rv_mag/mag_bin).astype('int')*mag_bin

    # -- Create light curves ---------------------------------------------------

    y_lc = []

    for i in range(n_samples):
        rv_mag[i] = 10. # !! tmp

        ind = np.nonzero(mags == rv_mag[i])[0]
        y = white_noise_amp[ind]*np.random.normal(size=len(t))
        rv_amp = loguniform.rvs(0, 1, size=rv_npeak[i])
        rv_period = loguniform.rvs(1/max_freq, 1/min_freq, size=rv_npeak[i])

        for j in range(rv_npeak[i]):
            y += np.sin( (2*np.pi/rv_period[j]) * t )

        y_lc.append(y)

        pdb.set_trace()

    return y_lc

    # >> https://heasarc.gsfc.nasa.gov/docs/tess/observing-technical.html

# -- Light curve and LS periodogram statistics ---------------------------------

def calc_stats(datapath, timescale, savepath):
    from scipy.stats import skew, kurtosis

    stats_desc = ['mean', 'stdev', 'skew', 'kurtosis']

    lcpath = datapath+'clip/'
    lspmpath = datapath+'timescale-'+str(timescale)+'sector/lspm/'
    txtpath = datapath+'timescale-'+str(timescale)+'sector/'
    ticid = [int(f[:-5]) for f in os.listdir(lspmpath)]

    statpath = savepath+'timescale-'+str(timescale)+'sector/feat_eng/S1-26/'
    dt.create_dir(statpath)
    dt.create_dir(statpath+'lspm-feats/')
    dt.create_dir(statpath+'all-feats/')

    lc_feat_desc = ['y_avg','y_std','y_skew','y_kur']
    lspm_feat_desc = ['Npeak']
    for feat in ['P', 'Pratio', 'A', 'Aratio']:
        for stat in ['avg', 'std', 'skew', 'kur']:
            lspm_feat_desc.append(feat+'_'+stat)
    
    with open(txtpath+'feat_eng.txt', 'w') as f:
        f.write('TICID,'+','.join(lc_feat_desc)+','+','.join(lspm_feat_desc)+'\n')

    stats = []
    for i in range(len(ticid)):
        
        lc_feats = []

        # >> load light curve
        t, y = get_lc(datapath+'clip/', ticid[i], timescale, rmv_nan=True,
                      norm=False)

        # >> calculate light curve statistics
        lc_feats.append(np.mean(y))
        lc_feats.append(np.std(y))
        lc_feats.append(skew(y))
        lc_feats.append(kurtosis(y))

        # >> load LS periodogram 
        hdul = fits.open(lspmpath+str(ticid[i])+'.fits')
        frequency = hdul[1].data['FREQ']
        power = hdul[1].data['LSPM']

        # -- calculate LS periodogram statistics -------------------------------
        t, y = get_lc(datapath+'clip/', ticid[i], timescale, rmv_nan=True,
                      norm=True, method='standardize')
        periods, lspm_feats = calc_lspm_stats(t=t, y=y, frequency=frequency,
                                              power=power, datapath=datapath,
                                              ticid=ticid[i],
                                              timescale=timescale,
                                              output_dir=statpath+'lspm-feats/')
        
        # -- save statistics ---------------------------------------------------

        with open(txtpath+'feat_eng.txt', 'a') as f:
            f.write(str(ticid[i])+','+\
                    ','.join([str(feat) for feat in lc_feats])+\
                    ','.join([str(feat) for feat in lspm_feats])+'\n')

        # -- plot light curve and LS-periodogram -------------------------------
        if len(periods) == 0:
            fig, ax = plt.subplots(2, 1, figsize=(8, 11))
        else:
            fig, ax = plt.subplots(3, 1, figsize=(8, 15))
        title = ''
        for j in range(len(lc_feats)):
            title+= lc_feat_desc[j]+': '+str(np.round(lc_feats[j],3))+','
        ax[0].set_title('Light curve for TIC '+str(ticid[i])+'\n'+title,
                        size='x-small')
        plot_lc(ax[0], t, y, marker='.', c='k', ms=2, linestyle='')

        title, line = '', 'Npeak: '+str(lspm_feats[0])+','
        for j in range(len(lspm_feats[1:])):
            if j%4 == 0:
                title+=line+'\n'
                line=''            
            line += lspm_feat_desc[j]+': '+str(np.round(lspm_feats[j],3))+','
        ax[1].set_title('LS-periodogram for TIC '+str(ticid[i])+'\n'+title,
                        size='x-small')
        plot_lspm(ax[1], frequency, power)

        if len(periods) > 0:
            folded_t, folded_y = calc_phase_curve(periods[0], t, y, frequency,
                                                  power)
            ax[2].set_title('Phase curve for TIC '+str(ticid[i])+'\n'+\
                            'period: '+str(np.round(periods[0], 3)),
                            size='x-small')
            plot_lc(ax[2], folded_t, folded_y, c='k', marker='.', ms=2, 
                    fillstyle='full', label='data', linestyle='')
            ax[2].set_xlabel('Time from midpoint epoch [days]')

        fig.tight_layout()
        out_f = statpath+'all-feats/feats_TIC'+str(ticid[i])+'.png'
        fig.savefig(out_f)
        print('Saved '+out_f)
        plt.close(fig)

def load_stats(mg, timescale=1):
    fname = mg.datapath+'timescale-'+str(timescale)+'sector/feat_eng.txt'
    # !! TODO: read cols from first line in fname
    cols = ['TICID', 'y_avg', 'y_std', 'y_skew', 'y_kur', 'Npeak', 'P_avg',
            'P_std', 'P_skew', 'P_kur', 'Pratio_avg', 'Pratio_std',
            'Pratio_skew', 'Pratio_kur', 'A_avg', 'A_std', 'A_skew',
            'A_kur', 'Aratio_avg', 'Aratio_std', 'Aratio_skew', 'Aratio_kur']
    filo = np.loadtxt(fname, skiprows=1, delimiter=',')
    inds = np.nonzero(~np.isnan(filo[:,-1]))

    print('Periodic: '+str(len(inds[0]))+' / '+str(len(filo)))

    mg.objid = filo[inds][:,0]
    mg.x_train = filo[inds][:,1:]

        
def plot_lspm(ax, freq, lspm, plot_freq=False, **kwargs):
    if plot_freq:
        ax.plot(freq, lspm, **kwargs)
        ax.set_xlabel('Frequency [days^-1]')
    else:
        ax.plot(1/freq, lspm, **kwargs)
        ax.set_xlabel('Period [days]')
    ax.set_ylabel('Power')
    ax.set_yscale('log')
    return ax

def plot_lc(ax, time, flux, **kwargs):
    # if 'linestyle' in kwargs.keys():
    #     ax.plot(time, flux, **kwargs)
    # else:
    #     ax.plot(time, flux, marker='.', c='k', ms=2, fillstyle='full',
    #             **kwargs)
    if len(kwargs.keys()) == 0:
        ax.plot(time, flux, marker='.', c='k', ms=2, fillstyle='full',
                linestyle='')
    else:
        ax.plot(time, flux, **kwargs)
    ax.set_xlabel('Time [BJD - 2457000]')
    ax.set_ylabel('Relative Flux')

