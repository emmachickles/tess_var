"""
Created on 210919

run_preprocessing.py
Script to perform preprocessing on SPOC light curves, with the science focus of
clustering and identifying anomalies of periodic and quasiperiodic varaibility.
@author: Emma Chickles (emmachickles)

Run with:
$ python tess_stellar_var/run_preprocessing.py sectors_num1 sectors_num2 ...

If given no arguments, will run all sectors:
$ python tess_stellar_var/run_preprocessing.py

"""

from __init__ import *
import feat_eng as fe

# >> what to run?
prep_spoc    = True  # >> preprocesses all short-cadence SPOC data
target_prep  = False # >> preprocess a single target
diag_only    = False # >> produce diagnostic plots

# >> inputs
n_sigma      = 7
plot         = True
setup        = '/home/echickle/work/tess_var/setup.txt'

# >> prepare specified targets (if target_prep = True)
# targets      = [31850842] # 
# >> Sectors 1 complex rotators reported in Zhan et al. 2019 
# targets      = [38820496, 177309964, 206544316, 234295610, 289840928,
#                 425933644, 425937691] # >> known complex rotators in sectors 1
targets = [30265037, 44736746]

# ------------------------------------------------------------------------------

print(sys.argv)
if len(sys.argv) > 1:
    sectors = []
    for arg in sys.argv[1:]:
        sectors.append('sector-%02d'%int(float(arg)))
else :
    sectors = []
    for s in range(1,27):
        sectors.append('sector-%02d'%int(float(s)))        
print('Preprocessing the following sectors: ')
print(sectors)

# ------------------------------------------------------------------------------

# >> Initialize Mergen object
mg = mergen(setup=setup)

# >> Perform preprocessing for all SPOC data 
if prep_spoc:
    # mg.clean_data() # >> quality mask
    # fe.sigma_clip_data(mg) # >> sigma clip
    # fe.calc_lspm_mult_sector(mg, sectors=sectors, timescale=1)
    fe.preprocess_lspm(mg, timescale=1)
    pdb.set_trace()

    # >> short timescale (1 month)
    # fe.calc_lspgram_avg_sector(mg, sectors=sectors, overwrite=True) 
    # fe.preprocess_lspgram(mg, timescale=1)
    # fe.calc_stats(datapath, timescale=1, savepath=savepath)


    # fe.test_simulated_data(savepath, datapath, timescale=1)

    # >> medium timescale (6 months, 19366 targets in S1-26 2-min)
    fe.calc_lspgram_mult_sector(mg, sectors=sectors, timescale=6)
    fe.preprocess_lspgram(mg, timescale=6, n_chunk=2)

    # >> long timescale (12 months, 3000 targets in S1-26 2-min)
    fe.calc_lspgram_mult_sector(mg, sectors=sectors, timescale=13,
                                   plot_int=50)
    fe.preprocess_lspgram(mg, timescale=13, n_chunk=2)

# >> Alternatively perform quality and sigma-clip masking for a single target
if target_prep:
    for target in targets:
        # savepath = '/scratch/echickle/tmp/'
        # lcfile = mg.datapath+'mask/sectors-01/'+str(target)+'.fits'
        # fe.sigma_clip_lc(mg, lcfile, plot=plot, n_sigma=n_sigma,
        #                  savepath=savepath)

        # data,meta=dt.open_fits(fname=lcfile)
        # t, y = data['TIME'], data['FLUX']
        # fe.calc_ls_pgram(mg, lcfile, plot=True, savepath=savepath)
        # feat=fe.calc_phase_curve(t,y,plot=plot,output_dir='/home/echickle/tmp/',
        #                          prefix='TIC'+str(target)+'-')
        
        fe.calc_ls_pgram(mg, target, plot=True,
                      savepath='/scratch/echickle/tmp/detrend/',
                      lspmpath='/scratch/echickle/tmp/', timescale=1)

# >> Create some diagnostic plots for masking
if diag_only:
    fe.sigma_clip_diag(mg)


