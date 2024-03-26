# -*- coding: utf-8 -*-
"""
Pre-process BrainVision EEG files

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os
import mne
import numpy as np
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up paths

data_root = r'R:\DRS-PSR\Seb\EEG_2back\data'
deriv_root = r'R:\DRS-PSR\Seb\EEG_2back\derivatives'

# create paths
if not op.exists(deriv_root):
    os.makedirs(deriv_root)

# scanning session info
subject = "11766"

if not op.exists(op.join(deriv_root, subject)):
    os.makedirs(op.join(deriv_root, subject))
    
#%% load data and events

data_fname = subject + "_nback"
data = mne.io.read_raw_brainvision(op.join(data_root, subject, 
                                           data_fname + '.vhdr'), preload=True)
data.info
orig_freq = data.info['sfreq']
data_raw = data.copy()  # these lines are added to maintain a copy of the data
                        # at each step - useful for debugging.
                        
### events from marker file
events = mne.events_from_annotations(data)[0]
mne.viz.plot_events(events)

#%% make appropriate montage

montage = mne.channels.make_standard_montage("easycap-M1")
montage.plot()
data.set_montage(montage, on_missing="ignore")

#%% downsample

sfreq = 500
data.resample(sfreq=sfreq)

### manually downsample events
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))

#%% separate out ECG for later

ECG = data.copy().pick_channels(["ECG"])
data.drop_channels("ECG") # don't need ECG in main data object

#%% automated bad channel removal using noise covariance

# get covariance of high frequency (most likely domain to be just noise)
noise = data.copy().pick('eeg').filter(50, 100)
noise_cov = mne.compute_raw_covariance(noise)

# now take diagonal of noise covariance
noise_cov_diag = noise_cov.as_diag().data
mean_noise = np.mean(noise_cov_diag)
std_noise = np.std(noise_cov_diag)
#bad_chan_i = noise_cov_diag - mean_noise > std_noise
bad_chan_i = noise_cov_diag > 2*mean_noise
bad_chan_i = np.arange(len(noise.ch_names))[bad_chan_i]
bad_chan = np.array(noise.ch_names)[bad_chan_i]

# plot 
plt.figure()
plt.plot(noise_cov_diag)
plt.hlines(2*mean_noise, xmin=0, xmax=len(noise.ch_names), color='black')
plt.xlabel("Channel Number")
plt.legend(["Noise Cov Diagonal", "Reject"])
plt.scatter(bad_chan_i, noise_cov_diag[bad_chan_i], color='red')

for i, txt in enumerate(bad_chan):
    plt.annotate(txt, (bad_chan_i[i], noise_cov_diag[bad_chan_i[i]]))

# set bads
data.info['bads'] = list(bad_chan)

# interpolate
data.interpolate_bads()

#%% broadband filter

data.filter(l_freq=1, h_freq=45)

#%% set average reference

data.set_eeg_reference('average', projection=True)

#%% plot PSD

data.plot_psd(fmax=45, picks='eeg').show()

#%% annotate muscle artefacts etc (DONT USE ZSCORE HERE)

muscle_annot, bad_chan = mne.preprocessing.annotate_amplitude(
                                    data, peak=dict(eeg=40e-6), picks='eeg',
                                    bad_percent=5, min_duration=1/sfreq)
muscle_annot.onset = muscle_annot.onset - 0.4
muscle_annot.duration = muscle_annot.duration + 0.4

data.set_annotations(muscle_annot)
data.plot()

#%% Automated ICA removal

ecg_data = data.copy().add_channels([ECG], force_update_info=True)
blink_channels = ['Fpz', 'Fp1', 'Fp2'];

ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data.copy().pick('eeg'), reject_by_annotation=True)

blink_indices, blink_scores = ica.find_bads_eog(data, blink_channels,
                                                reject_by_annotation=True)
ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_data, 'ECG', reject_by_annotation=True)

ica.exclude = list(dict.fromkeys(blink_indices + ecg_indices))

# plot diagnostics
ica.plot_properties(data, picks=ica.exclude)
ica.plot_sources(data)

# apply
ica.apply(data)

#%% automated bad (2 s) segment repair

duration = 2 + 1/sfreq
fake_events = mne.make_fixed_length_events(data, duration=duration)

event_id = 1
tmin, tmax = 0, 2
fake_epochs = mne.Epochs(
    data,
    fake_events,
    event_id,
    tmin,
    tmax,
    reject=dict(eeg=100e-6),
    baseline=None,
    preload=True
    )

dropped = [n for n, dl in enumerate(fake_epochs.drop_log) if len(dl)] 

# data to be fixed
fixed_epochs = mne.Epochs(
    data,
    fake_events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    reject=None,
    preload=True,
    reject_by_annotation=False
    )

fixed_data = fixed_epochs.get_data()

# replace with adjacent epoch
for bad_epoch in dropped:
    if bad_epoch > 0:
        epoch_replacement = fake_epochs[bad_epoch-1].get_data()
    else: 
        epoch_replacement = fake_epochs[bad_epoch+1].get_data()        
    fixed_data[bad_epoch,:,:] = epoch_replacement
    
# now make new Epochs object with fixed data
fixed_info = fixed_epochs.info
fixed_epochs = mne.EpochsArray(fixed_data, fixed_info)

# convert back into raw object
df = fixed_epochs.to_data_frame()
names = df.columns.to_list()[3:]
fixed_data = df.iloc[:,3:].to_numpy().transpose()/1e6  # go back to volts
data = mne.io.RawArray(fixed_data, data.info)
data.plot()

#%% make noise covariance from all data filtered into higher range

noise = data.copy().filter(35, 45)

#%% save out

data.save(op.join(deriv_root, subject, subject + "_nback_data.fif"), overwrite=True)
noise.save(op.join(deriv_root, subject, subject + "_nback_noise.fif"), overwrite=True)
mne.write_events(op.join(deriv_root, subject, subject + "_nback_events.fif"), 
                 events, overwrite=True)
