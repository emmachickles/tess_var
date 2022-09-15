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

start = datetime.now()
fe.prep_batch(datapath='/scratch/submit/tess/data/',
              savepath='/data/submit/echickle/data/timescale-1sector/')
end = datetime.now()
dur_sec = (end-start).total_seconds()
print(dur_sec)
pdb.set_trace()

# >> Initialize Mergen object
mg = mergen(setup=setup)

# >> Perform preprocessing for all SPOC data 
# mg.clean_data() # >> quality mask
# fe.sigma_clip_data(mg) # >> sigma clip
# fe.calc_lspm_mult_sector(mg, sectors=sectors, timescale=1)
# fe.preprocess_lspm(mg, timescale=1)
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
