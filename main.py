"""
210402 - Emma Chickles
"""

from __init__ import *

datapath = '/nfs/blender/data/tdaylan/data/'
savepath = '/nfs/blender/data/tdaylan/data/'
datatype = 'SPOC'
mdumpcsv = datapath + 'Table_of_momentum_dumps.csv'

obj = mg.mergen(datapath, savepath, datatype, mdumpcsv)

