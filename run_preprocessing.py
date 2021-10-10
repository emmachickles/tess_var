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
metapath     = '/scratch/data/tess/meta/'
datatype     = 'SPOC'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
sector       = 1
n_sigma      = 7
plot         = True
# targets      = [31850842] # >> for target_prep
targets      = [38820496, 177309964, 206544316, 234295610, 289840928,
                425933644, 425937691] # >> known complex rotators in sector 1

# >> what to run?
prep_spoc    = False
target_prep  = False # >> preprocess a single target
diag_only    = True # >> produce diagnostic plots

# ------------------------------------------------------------------------------

# >> Initialize Mergen object
mg = mergen.mergen(datapath, savepath, datatype, metapath=metapath,
                   mdumpcsv=mdumpcsv)

# >> Perform preprocessing for all SPOC data 
if prep_spoc:
    # mg.clean_data() # >> quality mask
    # fe.sigma_clip_data(mg)
    fe.compute_ls_pgram_data(mg)
    # fe.load_lspgram(mg)

# >> Alternatively perform quality and sigma-clip masking for a single target
if target_prep:
    for target in targets:

        savepath = '/scratch/echickle/tmp/'
        lcfile = mg.datapath+'mask/sector-01/'+str(target)+'.fits'
        # fe.sigma_clip_lc(mg, lcfile, plot=plot, n_sigma=n_sigma,
        #                  savepath=savepath)
        data,meta=dt.open_fits(fname=lcfile)
        t, y = data['TIME'], data['FLUX']
        fe.compute_ls_pgram(mg, lcfile, plot=True, n_freq=200000, savepath=savepath)
        # feat=fe.calc_phase_curve(t,y,plot=plot,output_dir='/home/echickle/tmp/',
        #                          prefix='TIC'+str(target)+'-')


# >> Create some diagnostic plots for masking
if diag_only:
    fe.sigma_clip_diag(mg)

# # >> Compute LS Periodograms
# if sector_prep:
#     dt.compute_ls_pgram_sector(datapath, sector, plot=True,
#                                savepath=mg.ensbpath)


