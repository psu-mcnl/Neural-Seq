import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
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
from scipy import interpolate
import pickle
import copy
import random

import seaborn as sns
import warnings
from tqdm import tqdm
import json

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed

from settings import settings
from utils import *
from filters import filters

cache = EcephysProjectCache.from_warehouse(manifest=settings.neuropixels.manifest_path)
session_fc_ids = settings.neuropixels.session_fc_ids

def movie_timefield_spike_rates(sess_id, resume=False):

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'MovSpikeData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        MovSpikeData = pickle.load(handle)
        
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'jsonDataInfo'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath) as handle:
        jsonDataInfo = json.load(handle)

    varName = 'MovTimeFieldsData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        MovTimeFieldsData = pickle.load(handle)

    spike_cnt_per_repeat = MovSpikeData.spike_cnt_per_repeat.values.astype(float)
    spike_cnt_per_repeat[MovSpikeData.invalid] = np.nan
    spike_cnt_per_repeat = xr.DataArray(spike_cnt_per_repeat, 
                                        coords=MovSpikeData.spike_cnt_per_repeat.coords)

    stimIDs = spike_cnt_per_repeat.stimulus_presentation_id.values
    CA1_unit_ids = SessData.units[SessData.units.ecephys_structure_acronym.str.match('CA1')].index.values

    preRest_stim_ids = stimIDs[MovSpikeData.stimTable.loc[stimIDs].stimulus_block == 3]
    postRest_stim_ids = stimIDs[MovSpikeData.stimTable.loc[stimIDs].stimulus_block == 8]
    preRest_spike_cnts = spike_cnt_per_repeat.sel(unit_id=CA1_unit_ids, stimulus_presentation_id=preRest_stim_ids)
    postRest_spike_cnts = spike_cnt_per_repeat.sel(unit_id=CA1_unit_ids, stimulus_presentation_id=postRest_stim_ids)

    """ pre-rest movie """
    timefield_bins = np.arange(0, 30+0.1, 0.5)
    n_repeats = preRest_spike_cnts.shape[0]
    t = preRest_spike_cnts.time_relative_to_stimulus_onset.values
    t = np.repeat(t.reshape(1,-1), n_repeats, axis=0).reshape(-1)
    spike_cnts = preRest_spike_cnts.values.reshape(-1,CA1_unit_ids.shape[0]).T

    rtns = stats.binned_statistic(t, spike_cnts, statistic=np.nansum, bins=timefield_bins)
    spike_hist = rtns.statistic
    spike_hist = (spike_hist.T / spike_hist.sum(axis=1)).T
    preRest_spike_hist = spike_hist

    # spike count for time field for all repeat
    spike_cnts_tf = []
    for spike_cnts_1repeat in preRest_spike_cnts:
        t = spike_cnts_1repeat.time_relative_to_stimulus_onset.values
        rtns = stats.binned_statistic(t, spike_cnts_1repeat.values.T, statistic=np.nansum, bins=timefield_bins)
        spike_cnts_tf.append(rtns.statistic.T)
    spike_cnts_tf = np.stack(spike_cnts_tf, axis=0)
    preRest_spike_cnts_tf = spike_cnts_tf

    time_unit_mask = MovTimeFieldsData.movPreRest.positive.tf_mod_pvals < 0.05
    preRest_time_unit_ids = MovSpikeData.spike_cnt_per_repeat.unit_id.values[time_unit_mask]
    nontime_unit_mask = MovTimeFieldsData.movPreRest.positive.tf_mod_pvals > 0.1
    preRest_nontime_unit_ids = MovSpikeData.spike_cnt_per_repeat.unit_id.values[nontime_unit_mask]

    s = '[CA1 Pre-Rest Time neurons: {:d}]'.format(len(np.intersect1d(CA1_unit_ids, preRest_time_unit_ids)))
    s += '\n[CA1 Pre-Rest Non-Time neurons: {:d}]'.format(len(np.intersect1d(CA1_unit_ids, preRest_nontime_unit_ids)))

    """ post-rest movie """
    n_repeats = postRest_spike_cnts.shape[0]
    t = postRest_spike_cnts.time_relative_to_stimulus_onset.values
    t = np.repeat(t.reshape(1,-1), n_repeats, axis=0).reshape(-1)
    spike_cnts = postRest_spike_cnts.values.reshape(-1,CA1_unit_ids.shape[0]).T

    rtns = stats.binned_statistic(t, spike_cnts, statistic=np.nansum, bins=timefield_bins)
    spike_hist = rtns.statistic
    spike_hist = (spike_hist.T / spike_hist.sum(axis=1)).T
    postRest_spike_hist = spike_hist

    time_unit_mask = MovTimeFieldsData.movPostRest.positive.tf_mod_pvals < 0.05
    postRest_time_unit_ids = MovSpikeData.spike_cnt_per_repeat.unit_id.values[time_unit_mask]
    nontime_unit_mask = MovTimeFieldsData.movPostRest.positive.tf_mod_pvals > 0.1
    postRest_nontime_unit_ids = MovSpikeData.spike_cnt_per_repeat.unit_id.values[nontime_unit_mask]

    s += '\n[CA1 Post-Rest Time neurons: {:d}]'.format(len(np.intersect1d(CA1_unit_ids, postRest_time_unit_ids)))
    s += '\n[CA1 Post-Rest Non-Time neurons: {:d}]'.format(len(np.intersect1d(CA1_unit_ids, postRest_nontime_unit_ids)))
    print(s)
    
    """ pre-rest movie (shuffled activities) """
    rng = np.random.default_rng()
    s = preRest_spike_hist.shape
    preRest_spike_shuffled = rng.permutation(preRest_spike_hist.reshape(-1)).reshape(s)

    """ post-rest movie (shuffled activities) """
    s = postRest_spike_hist.shape
    postRest_spike_shuffled = rng.permutation(postRest_spike_hist.reshape(-1)).reshape(s)

    """ save .mat file """
    mat_dict = dict()
    mat_dict['preRestMov_spikeRates'] = preRest_spike_hist
    mat_dict['preRestMov_spikeRates_shuffled'] = preRest_spike_shuffled
    mat_dict['preRestMov_spikeRatesByRepeat'] = preRest_spike_cnts_tf
    mat_dict['preRest_time_unit_ids'] = preRest_time_unit_ids
    mat_dict['preRest_nontime_unit_ids'] = preRest_nontime_unit_ids

    mat_dict['postRestMov_spikeRates'] = postRest_spike_hist
    mat_dict['postRestMov_spikeRates_shuffled'] = postRest_spike_shuffled
    mat_dict['postRest_time_unit_ids'] = postRest_time_unit_ids
    mat_dict['postRest_nontime_unit_ids'] = postRest_nontime_unit_ids

    mat_dict['CA1_unit_ids'] = CA1_unit_ids
    mat_dict['timefield_bins'] = timefield_bins
    mat_dict['timefield'] = 0.5 * (timefield_bins[:-1] + timefield_bins[1:])
    
    fpath = jsonDataInfo['replay_decoding']['dat_path']['mov_spikeRate']
    print('save spike rate data to {:s}'.format(fpath))
    sio.savemat(fpath, mat_dict)

def rest_ripple_spikes(sess_id, resume=False):

    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'jsonDataInfo'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath) as handle:
        jsonDataInfo = json.load(handle)

    if resume and os.path.isfile(jsonDataInfo['replay_decoding']['dat_path']['rest_ripple_spikes']):
        return

    sessDir = settings.neuCascade.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'lfpData'
    fName = getattr(settings.neuCascade.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        lfpData = pickle.load(handle)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    spontTable = session_data.get_stimulus_table('spontaneous')
    spont_tstart = spontTable['start_time'].values[-2]
    spont_tstop = spontTable['stop_time'].values[-2]

    """ ripple events """
    chn = np.argmax([len(item) if isinstance(item, np.ndarray) or isinstance(item, list) else 0 \
                     for item in lfpData.ripplectrind])
    ripple_inds = lfpData.ripplectrind[chn]
    ripple_time = lfpData.CA1time[ripple_inds]
    ripple_time += spont_tstart

    ripple_time_filtered = []
    for t in np.sort(ripple_time):
        if len(ripple_time_filtered) != 0:
            if t - ripple_time_filtered[-1] < 0.15:
                continue
        ripple_time_filtered.append(t)
    ripple_time_filtered = np.array(ripple_time_filtered)

    """ CA1 spikes at ripples """
    CA1_unit_ids = SessData.units[SessData.units.ecephys_structure_acronym.str.match('CA1')].index.values
    CA1_rp_spikes = {}
    t_winsize = 0.3
    for rp_i, t_anchor in enumerate(tqdm(ripple_time_filtered, desc='ripple events')):
        win_st, win_ed = (t_anchor - t_winsize, t_anchor + t_winsize)
        
        rp_spikes = {}
        for unit_id in CA1_unit_ids:
            spike_times = session_data.spike_times[unit_id]
            dmsk = (spike_times >= win_st) & (spike_times <= win_ed)
            rp_spikes['u{:d}'.format(unit_id)] = list(spike_times[dmsk] - t_anchor)
        CA1_rp_spikes['rp_{:d}'.format(rp_i)] = rp_spikes

    """ save .mat file """
    mat_dict = dict()
    mat_dict['CA1_spikes'] = CA1_rp_spikes
    mat_dict['CA1_unit_ids'] = CA1_unit_ids
    mat_dict['time_win_size'] = t_winsize
    mat_dict['spont_tstart'] = spont_tstart
    mat_dict['spont_tstop'] = spont_tstop
    mat_dict['ripple_time'] = ripple_time_filtered

    fpath = jsonDataInfo['replay_decoding']['dat_path']['rest_ripple_spikes']
    print('save ripple CA1 spike data to {:s}'.format(fpath))
    sio.savemat(fpath, mat_dict)

def rest_mua_event_spikes(sess_id, resume=False):
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'jsonDataInfo'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath) as handle:
        jsonDataInfo = json.load(handle)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    spontTable = session_data.get_stimulus_table('spontaneous')
    spont_tstart = spontTable['start_time'].values[-2]
    spont_tstop = spontTable['stop_time'].values[-2]
    spont_tdur = spontTable['duration'].values[-2]
    spont_stimID = spontTable.index[-2]

    """ CA1 MUA events """
    spike_tRes = 0.02
    spike_tBins = np.arange(0, spont_tdur+spike_tRes/2, spike_tRes)
    CA1_unit_ids = SessData.units[SessData.units.ecephys_structure_acronym.str.match('CA1')].index.values

    spike_cnt = session_data.presentationwise_spike_counts(spike_tBins, spont_stimID, CA1_unit_ids).squeeze()
    spike_cnt = xr.DataArray(spike_cnt, dims=['time', 'unit_id'], 
                                        coords={'time': spike_cnt.time_relative_to_stimulus_onset.values + spont_tstart,
                                                'unit_id': spike_cnt.unit_id.values})
    spike_gls = spike_cnt.mean(axis=1)
    spike_cnt_n = (spike_cnt - spike_cnt.mean(axis=0)) / spike_cnt.std(axis=0)
    spike_gls = spike_cnt_n.mean(axis=1)
    spike_gls_f = signal.sosfiltfilt(filters.flt2.sos, spike_gls)
    cand_events_edges, _ = signal.find_peaks(-spike_gls_f)
    cand_events_edges = cand_events_edges[1:-1]

    cand_events_peaks, _ = signal.find_peaks(spike_gls_f)
    cand_events_peaks = cand_events_peaks[1:-1]
    cand_events_peaks_time = spike_cnt.time.values[cand_events_peaks]

    """ CA1 spikes at ripples """
    CA1_event_spikes = {}
    t_winsize = 0.3
    for event_i, t_anchor in enumerate(tqdm(cand_events_peaks_time, desc='MUA events')):
        win_st, win_ed = (t_anchor - t_winsize, t_anchor + t_winsize)

        event_spikes = {}
        for unit_id in CA1_unit_ids:
            spike_times = session_data.spike_times[unit_id]
            dmsk = (spike_times >= win_st) & (spike_times <= win_ed)
            event_spikes['u{:d}'.format(unit_id)] = list(spike_times[dmsk] - t_anchor)
        CA1_event_spikes['event_{:d}'.format(event_i)] = event_spikes

    """ save .mat file """
    mat_dict = dict()
    mat_dict['CA1_spikes'] = CA1_event_spikes
    mat_dict['CA1_unit_ids'] = CA1_unit_ids
    mat_dict['time_win_size'] = t_winsize
    mat_dict['spont_tstart'] = spont_tstart
    mat_dict['spont_tstop'] = spont_tstop
    mat_dict['mua_event_time'] = cand_events_peaks_time

    fpath = jsonDataInfo['replay_decoding']['dat_path']['rest_muaEvents_spikes']
    print('save MUA event CA1 spike data to {:s}'.format(fpath))
    sio.savemat(fpath, mat_dict)

def rest_mua_event_spikes(sess_id, resume=False):
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'jsonDataInfo'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath) as handle:
        jsonDataInfo = json.load(handle)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    spontTable = session_data.get_stimulus_table('spontaneous')
    spont_tstart = spontTable['start_time'].values[-2]
    spont_tstop = spontTable['stop_time'].values[-2]
    spont_tdur = spontTable['duration'].values[-2]
    spont_stimID = spontTable.index[-2]

    """ CA1 MUA events """
    spike_tRes = 0.02
    spike_tBins = np.arange(0, spont_tdur+spike_tRes/2, spike_tRes)
    CA1_unit_ids = SessData.units[SessData.units.ecephys_structure_acronym.str.match('CA1')].index.values

    spike_cnt = session_data.presentationwise_spike_counts(spike_tBins, spont_stimID, CA1_unit_ids).squeeze()
    spike_cnt = xr.DataArray(spike_cnt, dims=['time', 'unit_id'], 
                                        coords={'time': spike_cnt.time_relative_to_stimulus_onset.values + spont_tstart,
                                                'unit_id': spike_cnt.unit_id.values})
    spike_gls = spike_cnt.mean(axis=1)
    spike_cnt_n = (spike_cnt - spike_cnt.mean(axis=0)) / spike_cnt.std(axis=0)
    spike_gls = spike_cnt_n.mean(axis=1)
    spike_gls_f = signal.sosfiltfilt(filters.flt2.sos, spike_gls)
    cand_events_edges, _ = signal.find_peaks(-spike_gls_f)
    cand_events_edges = cand_events_edges[1:-1]

    cand_events_peaks, _ = signal.find_peaks(spike_gls_f)
    cand_events_peaks = cand_events_peaks[1:-1]
    cand_events_peaks_time = spike_cnt.time.values[cand_events_peaks]

    """ CA1 spikes at ripples """
    CA1_event_spikes = {}
    t_winsize = 0.3
    for event_i, t_anchor in enumerate(tqdm(cand_events_peaks_time, desc='MUA events')):
        win_st, win_ed = (t_anchor - t_winsize, t_anchor + t_winsize)

        event_spikes = {}
        for unit_id in CA1_unit_ids:
            spike_times = session_data.spike_times[unit_id]
            dmsk = (spike_times >= win_st) & (spike_times <= win_ed)
            event_spikes['u{:d}'.format(unit_id)] = list(spike_times[dmsk] - t_anchor)
        CA1_event_spikes['event_{:d}'.format(event_i)] = event_spikes

    """ save .mat file """
    mat_dict = dict()
    mat_dict['CA1_spikes'] = CA1_event_spikes
    mat_dict['CA1_unit_ids'] = CA1_unit_ids
    mat_dict['time_win_size'] = t_winsize
    mat_dict['spont_tstart'] = spont_tstart
    mat_dict['spont_tstop'] = spont_tstop
    mat_dict['mua_event_time'] = cand_events_peaks_time

    fpath = jsonDataInfo['replay_decoding']['dat_path']['rest_muaEvents_spikes']
    print('save MUA event CA1 spike data to {:s}'.format(fpath))
    sio.savemat(fpath, mat_dict)

def rest_top_mua_event_spikes(sess_id, resume=False):
    sessDir = settings.projectData.dir.sessions / 'session_{:d}'.format(sess_id)
    varName = 'SessData'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath, 'rb') as handle:
        SessData = pickle.load(handle)

    varName = 'jsonDataInfo'
    fName = getattr(settings.projectData.files.sessions, varName)
    fPath = sessDir / fName
    with open(fPath) as handle:
        jsonDataInfo = json.load(handle)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        session_data = cache.get_session_data(sess_id)
        stimTable = session_data.get_stimulus_table()

    spontTable = session_data.get_stimulus_table('spontaneous')
    spont_tstart = spontTable['start_time'].values[-2]
    spont_tstop = spontTable['stop_time'].values[-2]
    spont_tdur = spontTable['duration'].values[-2]
    spont_stimID = spontTable.index[-2]

    """ CA1 MUA events """
    spike_tRes = 0.001
    spike_tBins = np.arange(0, spont_tdur+spike_tRes/2, spike_tRes)
    CA1_unit_ids = SessData.units[SessData.units.ecephys_structure_acronym.str.match('CA1')].index.values

    spike_cnt = session_data.presentationwise_spike_counts(spike_tBins, spont_stimID, CA1_unit_ids).squeeze()
    spike_cnt = xr.DataArray(spike_cnt, dims=['time', 'unit_id'], 
                                        coords={'time': spike_cnt.time_relative_to_stimulus_onset.values + spont_tstart,
                                                'unit_id': spike_cnt.unit_id.values})
    spike_gls = spike_cnt.sum(axis=1)
    spike_gls_f = spike_gls.to_dataframe()
    spike_gls_f = spike_gls_f.rolling(window=50, win_type="gaussian", center=True).mean(std=15)
    spike_gls_f = spike_gls_f['spike_counts'].to_xarray()

    peaks_inds, _ = signal.find_peaks(spike_gls_f)
    gls_peaks = spike_gls_f[peaks_inds]
    m, s = (np.nanmean(spike_gls_f), np.nanstd(spike_gls_f))
    mua_event_inds = peaks_inds[(gls_peaks - m) > (3 * s)]
    cand_events_peaks_time = spike_gls.time.values[mua_event_inds]

    """ CA1 spikes at ripples """
    CA1_event_spikes = {}
    t_winsize = 0.3
    for event_i, t_anchor in enumerate(tqdm(cand_events_peaks_time, desc='MUA events')):
        win_st, win_ed = (t_anchor - t_winsize, t_anchor + t_winsize)

        event_spikes = {}
        for unit_id in CA1_unit_ids:
            spike_times = session_data.spike_times[unit_id]
            dmsk = (spike_times >= win_st) & (spike_times <= win_ed)
            event_spikes['u{:d}'.format(unit_id)] = list(spike_times[dmsk] - t_anchor)
        CA1_event_spikes['event_{:d}'.format(event_i)] = event_spikes

    """ save .mat file """
    mat_dict = dict()
    mat_dict['CA1_spikes'] = CA1_event_spikes
    mat_dict['CA1_unit_ids'] = CA1_unit_ids
    mat_dict['time_win_size'] = t_winsize
    mat_dict['spont_tstart'] = spont_tstart
    mat_dict['spont_tstop'] = spont_tstop
    mat_dict['mua_event_time'] = cand_events_peaks_time

    fpath = jsonDataInfo['replay_decoding']['dat_path']['rest_topMuaEvent_spikes']
    print('save MUA event CA1 spike data to {:s}'.format(fpath))
    sio.savemat(fpath, mat_dict)

def proc_movie_spike_rate(resume=False):
    sessions2proc = session_fc_ids
    # sessions2proc = [781842082]

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing movie spike rates for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        movie_timefield_spike_rates(sess_id, resume=resume)

def proc_rest_ripple_spikes(resume=False):
    sessions2proc = np.setdiff1d(session_fc_ids, settings.neuropixels.session_removed)

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing resting ripple CA1 spikes for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        rest_ripple_spikes(sess_id, resume=resume)

def proc_rest_mua_event_spikes(resume=False):
    sessions2proc = np.setdiff1d(session_fc_ids, settings.neuropixels.session_removed)

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing resting MUA event CA1 spikes for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        rest_mua_event_spikes(sess_id, resume=resume)

def proc_rest_top_mua_event_spikes(resume=False):
    sessions2proc = np.setdiff1d(session_fc_ids, settings.neuropixels.session_removed)

    for i, sess_id in enumerate(sessions2proc):
        s = 'processing resting TOP MUA event CA1 spikes for session {:d} [{:d}|{:d}]'
        print(s.format(sess_id, i+1, len(sessions2proc)))
        rest_top_mua_event_spikes(sess_id, resume=resume)

def main():

    proc_movie_spike_rate(resume=False)

    # proc_rest_ripple_spikes(resume=True)

    # proc_rest_mua_event_spikes(resume=True)

    # proc_rest_top_mua_event_spikes(resume=True)


if __name__ == "__main__":
    main()