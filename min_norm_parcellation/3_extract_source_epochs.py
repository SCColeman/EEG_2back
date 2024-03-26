# -*- coding: utf-8 -*-
"""
Localise oscillatory activity to parcellations.

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

event_id = 65  # trigger of interest
tmin, tmax = 0, 40
epochs = mne.Epochs(
    data,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(30, 40),
    preload=True,
    reject_by_annotation=True)

#%% make inverse operator

inv = make_inverse_operator(data.info, fwd, noise_cov)

#%% parcellation beamformer

# get labels from parcellation
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)  # for fsaverage
subjects_dir = op.dirname(fs_dir)    # for fsaverage
fs_subject = "fsaverage"
parc = "HCPMMP1_combined"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)
labels = labels[2:]

lambda2 = 1
stcs = apply_inverse_epochs(
    epochs, inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
    method="eLORETA"
)

label_ts = mne.extract_label_time_course(
    stcs, labels, inv["src"], return_generator=False
)
del stcs

#%% create source epochs object
 
n_epochs = (len(epochs))
epoch_len = np.shape(epochs[0])[2]
source_epochs_data = np.zeros((len(epochs), len(labels), np.shape(epochs[0])[2]))
for s, stc_epoch in enumerate(label_ts):
    source_epochs_data[s,:,:] = stc_epoch
    
#fake epoch object
ch_names=[labels[i].name for i in range(len(labels))]
epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')
source_epochs = mne.EpochsArray(source_epochs_data, epochs_info)

source_epochs.plot("all")

#%% plot any region 

Brain = mne.viz.get_brain_class()
brain = Brain("fsaverage",
            "both",
            "inflated",
            subjects_dir=subjects_dir,
            size=(800, 600),
            )
aud_label = [label for label in labels][0]
brain.add_annotation("HCPMMP1_combined")