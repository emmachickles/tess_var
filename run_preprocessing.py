"""
Created on 210919

run_preprocessing.py
Script to perform preprocessing on SPOC light curves, with the science focus of
clustering and identifying anomalies of complex rotators.
@author: Emma Chickles (emmachickles)

"""

from __init__ import *
import feat_eng as fe

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
savepath     = '/scratch/echickle/'
# savepath = '/scratch/echickle/tmp/'
datatype     = 'SPOC'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
sector       = 1
plot         = True
target       = 31850842 # >> for target_prep
com_rot      = [38820496, 177309964, 206544316, 234295610, 289840928,
                425933644, 425937691] # >> known complex rotators in sector 1
n_sigma      = 7

# >> what to run?
sector_prep  = False  # >> preprocess a sector
target_prep  = False # >> preprocess a single target
com_rot_prep = True # >> preproecesses known complex rotators in Sector 1
diag_only    = False # >> produce diagnostic plots

# ------------------------------------------------------------------------------

# >> Initialize Mergen object
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   mdumpcsv=mdumpcsv)


# >> Perform quality and sigma-clip masking for a sector
if sector_prep:
    # mg.clean_data()
    # fe.sigma_clip_sector(mg, plot=plot, n_sigma=n_sigma)
    # fe.compute_ls_pgram_sector(mg, plot=plot)
    dt.DAE_preprocessing(datapath+'lspm/sector-01/')

# >> Alternatively perform quality and sigma-clip masking for a single target
if target_prep:
    lcfile = str(target)+'.fits'
    # fe.sigma_clip_lc(mg, lcfile, plot=plot, n_sigma=n_sigma)    

    t,y,meta=dt.open_fits(mg.datapath+'clip/sector-01/', fname=lcfile)
    feat=fe.calc_phase_curve(t,y,plot=plot,output_dir='/home/echickle/tmp/',
                             prefix='TIC'+str(target)+'-')

if com_rot_prep:
    for target in com_rot:

        lcfile = str(target)+'.fits'
        # fe.sigma_clip_lc(mg, lcfile, plot=plot, n_sigma=n_sigma)
        data,meta=dt.open_fits(mg.datapath+'clip/sector-01/', fname=lcfile)
        t, y = data['TIME'], data['FLUX']
        feat=fe.calc_phase_curve(t,y,plot=plot,output_dir='/home/echickle/tmp/',
                                 prefix='TIC'+str(target)+'-')


# >> Create some diagnostic plots for masking
if diag_only:
    dt.clean_sector_diag(datapath+'mask/sector-01/', mg.ensbpath, sector,
                         mdumpcsv)

# >> Compute LS Periodograms
if sector_prep:
    dt.compute_ls_pgram_sector(datapath, sector, plot=True,
                               savepath=mg.ensbpath)


