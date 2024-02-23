import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
from multiprocessing import Pool, current_process
import seaborn as sns
import warnings

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed

from settings import settings
from utils import *

cache = EcephysProjectCache.from_warehouse(manifest=settings.neuropixels.manifest_path)
session_fc_ids = settings.neuropixels.session_fc_ids

def define_neural_ensembles(sess_id, resume=False):

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    """ define neuron ensembles """
    neuEsmbl = Ensembles()

    esmbl = DataContainer()
    esmbl.unit_mask = np.full([SessData.unit_structure_acronym.shape[0]], True)
    esmbl.notes = 'all neurons'
    esmbl.name = 'All'
    esmbl_id = 1
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('CA1').values
    esmbl.notes = 'CA1 neurons'
    esmbl.name = 'CA1'
    esmbl_id = 2
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('CA2').values
    esmbl.notes = 'CA2 neurons'
    esmbl.name = 'CA2'
    esmbl_id = 3
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('CA3').values
    esmbl.notes = 'CA3 neurons'
    esmbl.name = 'CA3'
    esmbl_id = 4
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('DG').values
    esmbl.notes = 'DG neurons'
    esmbl.name = 'DG'
    esmbl_id = 5
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('SUB').values
    esmbl.notes = 'SUB neurons'
    esmbl.name = 'SUB'
    esmbl_id = 6
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    HP_regions = ['CA1', 'CA2', 'CA3', 'DG', 'SUB']
    unit_mask = np.full([SessData.unit_structure_acronym.shape[0]], False)
    for area in HP_regions:
        unit_mask |= SessData.unit_structure_acronym.str.match(area).values
    esmbl.unit_mask = unit_mask
    esmbl.notes = 'Hippocampal neurons'
    esmbl.name = 'Hippo'
    esmbl_id = 7
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.unit_structure_acronym.str.match('VIS').values
    esmbl.notes = 'VIS neurons'
    esmbl.name = 'VIS'
    esmbl_id = 8
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.units.ecephys_structure_acronym.isin(['VISp']).values
    esmbl.notes = 'VISp neurons'
    esmbl.name = 'VISp'
    esmbl_id = 9
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.units.ecephys_structure_acronym.isin(['LGd']).values
    esmbl.notes = 'LGd neurons'
    esmbl.name = 'LGd'
    esmbl_id = 10
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    esmbl = DataContainer()
    esmbl.unit_mask = SessData.units.ecephys_structure_acronym.isin(['LP']).values
    esmbl.notes = 'LP neurons'
    esmbl.name = 'LP'
    esmbl_id = 11
    neuEsmbl.register_esmbl(esmbl_id, esmbl)

    var2save = ['neuEsmbl']
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    for varName in var2save:
        fName = getattr(settings.projectData.files.sessions, varName)
        fPath = sessDir / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def proc_neural_ensembles(resume=False):
    sessions2proc = session_fc_ids

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing neural ensembles for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        define_neural_ensembles(sess_id, resume=resume)

def main():

    proc_neural_ensembles(resume=False)


if __name__ == "__main__":
    main()