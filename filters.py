import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy import linalg
from scipy import stats
import scipy.io as sio
import os
import shutil
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pickle
import copy
import random
import seaborn as sns

from settings import settings
from utils import *

filters = DataContainer()

""" filter-1 """
Wn = 2
N = 3
fs = 50
btype = 'lowpass'
sos = signal.butter(N, Wn, fs=fs, btype=btype, output='sos')
flt = DataContainer()
flt.Wn = Wn
flt.N = N
flt.sos = sos
flt.fs = fs
flt.btype = btype
flt.description = 'low pass filters for spike data (replay identification) (50Hz)'
filters.flt1 = flt

""" filter-2 """
Wn = 4
N = 3
fs = 50
btype = 'lowpass'
sos = signal.butter(N, Wn, fs=fs, btype=btype, output='sos')
flt = DataContainer()
flt.Wn = Wn
flt.N = N
flt.sos = sos
flt.fs = fs
flt.btype = btype
flt.description = 'low pass filters for spike data (replay identification) (50Hz)'
filters.flt2 = flt

""" filter-3 """
Wn = 6
N = 3
fs = 50
btype = 'lowpass'
sos = signal.butter(N, Wn, fs=fs, btype=btype, output='sos')
flt = DataContainer()
flt.Wn = Wn
flt.N = N
flt.sos = sos
flt.fs = fs
flt.btype = btype
flt.description = 'low pass filters for spike data (replay identification) (50Hz)'
filters.flt3 = flt

""" filter-4 """
Wn = 8
N = 3
fs = 50
btype = 'lowpass'
sos = signal.butter(N, Wn, fs=fs, btype=btype, output='sos')
flt = DataContainer()
flt.Wn = Wn
flt.N = N
flt.sos = sos
flt.fs = fs
flt.btype = btype
flt.description = 'low pass filters for spike data (replay identification) (50Hz)'
filters.flt4 = flt

""" filter-5 """
Wn = 10
N = 3
fs = 50
btype = 'lowpass'
sos = signal.butter(N, Wn, fs=fs, btype=btype, output='sos')
flt = DataContainer()
flt.Wn = Wn
flt.N = N
flt.sos = sos
flt.fs = fs
flt.btype = btype
flt.description = 'low pass filters for spike data (replay identification) (50Hz)'
filters.flt5= flt