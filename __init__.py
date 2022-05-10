"""
210402 - Emma Chickles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import os
import gc
import pdb
import sys

import astropy.coordinates as coord
import astropy.units as u
from astroquery.simbad import Simbad

# >> add working directory to path (assumes mergen, ephesus in wd)
sys.path.append(os.getcwd())

from mergen import mergen # >> library for unsupervised learning in astro
from mergen import learn_utils as lt
from mergen import plot_utils  as pt
from mergen import data_utils  as dt

# from mergen.mergen import learn_utils as lt
# from mergen.mergen import plot_utils as pt
# from mergen.mergen import data_utils as dt
# from mergen.mergen import mergen


