# -*- coding: utf-8 -*-
"""
Localise oscillatory activity during 2-back blocks.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse_cov
import numpy as np
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up paths

data_root = r'R:\DRS-PSR\Seb\EEG_2back\data'
deriv_root = r'R:\DRS-PSR\Seb\EEG_2back\derivatives'

# scanning session info
subject = "11766"

# frequency band info
band = "alpha"
bplims = (13, 30)

# options
plot_brains = True
plot_timecourse = True
plot_TFS = True

#%% load data

data = mne.io.Raw(op.join(deriv_root, subject, subject + "_nback_data.fif"), 
           preload=True)
noise = mne.io.Raw(op.join(deriv_root, subject, subject + "_nback_noise.fif"), 
           preload=True)
events = mne.read_events(op.join(deriv_root, subject, subject + "_nback_events.fif"))
fwd = mne.read_forward_solution(op.join(deriv_root, subject, subject + "_nback_fwd.fif"))

sfreq = data.info['sfreq']

#%% compute noise covariance

noise_cov = mne.compute_raw_covariance(noise, reject_by_annotation=True)

#%% epoch

event_id = 96  # trigger of interest
tmin, tmax = -0.5, 1.5
epochs = mne.Epochs(
    data,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(-0.5, -0.2),
    preload=True,
    reject_by_annotation=True)

#%% make inverse operator

inv = make_inverse_operator(data.info, fwd, noise_cov)

#%% get pseudo T

# filter epochs
epochs_filt = epochs.copy().filter(bplims[0], bplims[1])
lambda2 = 1  # this should be 1/SNR^2, but we assume SNR=1 for non-evoked data

# active and control covariance of filtered data
act_min, act_max = 0.6, 1
con_min, con_max = -0.4, 0

active_cov = mne.compute_covariance(epochs_filt, tmin=act_min, tmax=act_max, 
                                    method="shrunk")
control_cov= mne.compute_covariance(epochs_filt, tmin=con_min, tmax=con_max, 
                                    method="shrunk")

# right pseudo T
stc_active = apply_inverse_cov(
    active_cov, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
    method="eLORETA"
    )
stc_base = apply_inverse_cov(
    control_cov, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
    method="eLORETA"
    ) 
pseudoT = (stc_active - stc_base) / (stc_active + stc_base)

# for plotting
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)  # for fsaverage
subjects_dir = op.dirname(fs_dir)    # for fsaverage
fs_subject = "fsaverage"

if plot_brains:
    pseudoT.plot(src=fwd['src'], subject=fs_subject,
                subjects_dir=subjects_dir,
                surface="inflated",
                views=["lat", "med"],
                size=600,
                hemi="split",
                smoothing_steps=20,
                time_viewer=False,
                show_traces=False,
                colorbar=True)
        
# save pseudo T
pseudoT_fname = subject + "_nback_pseudoT_button"
pseudoT.save(op.join(deriv_root, subject, pseudoT_fname), overwrite=True)

#%% extract peak timecourse

# create generator
stc_epochs = apply_inverse_epochs(
    epochs, inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
    method="eLORETA"
)

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label_list = [32, 44, 48]
hemi = "lh"
label_name = hemi + " motor"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!

# combine vertices and pos from labels
vertices = []
pos = []
for l in label_list:
    vertices.append(labels[l].vertices)
    pos.append(labels[l].pos)
vertices = np.concatenate(vertices, axis=0)   
pos = np.concatenate(pos, axis=0)

# sort vertices and pos
vert_order = np.argsort(vertices)
vertices_ordered = vertices[vert_order]
pos_ordered = pos[vert_order,:]

new_label = mne.Label(vertices_ordered, pos_ordered, hemi=hemi, 
                      name=label_name, subject="fsaverage")

stc_inlabel = pseudoT.in_label(new_label)
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
n_epochs = len(epochs)
epoch_len = np.shape(epochs[0])[2]
epoch_peak_data = np.zeros((n_epochs,1,epoch_len))
for s,stc_epoch in enumerate(stc_epochs):
    stc_epoch_label = mne.extract_label_time_course(stc_epoch, new_label, 
                                                    fwd['src'], mode=None)
    epoch_peak_data[s,0,:] = stc_epoch_label[0][label_peak,:]
    
# make source epoch object
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=epochs.info["sfreq"],
                              ch_types=ch_types)
source_epochs = mne.EpochsArray(epoch_peak_data, source_info,
                                tmin=epochs.tmin)

# TFR
if plot_TFS:
    baseline = (-0.4, -0.1)
    freqs = np.arange(1,35)
    n_cycles = freqs/2
    power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                               use_fft=True, picks="all"
                                               )
    power[0].plot(picks="all", baseline=baseline)

# timecourse
if plot_timecourse:
    source_epochs_filt = source_epochs.copy().filter(bplims[0], bplims[1], picks="all")
    source_epochs_hilb = source_epochs_filt.copy().apply_hilbert(envelope=True, picks="all")
    peak_timecourse = source_epochs_hilb.average(picks="all").apply_baseline(baseline)
    
    plt.figure()
    plt.plot(peak_timecourse.times, peak_timecourse.get_data()[0], color="black")
    plt.ylabel("Oscillatory Power (A.U)")
    plt.xlabel("Time (s)")
    plt.title(new_label.name)

