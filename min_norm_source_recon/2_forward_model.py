# -*- coding: utf-8 -*-
"""
Create forward model based on FreeSurfer reconstruction with 
EEG BrainVision files.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import mne
import pandas as pd

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

#%% Get FS reconstruction for subject or use fsaverage for quick testing

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)  # for fsaverage
subjects_dir = op.dirname(fs_dir)    # for fsaverage
fs_subject = "fsaverage"

plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200])

mne.viz.plot_bem(**plot_bem_kwargs)

#%% set proper montage using .pos file, CREATED FROM AN EINSCAN FILE, MODIFY ACCORDINGLY

elec_names = data.ch_names

# load in pos file to pandas dataframe
df = pd.read_table(op.join(data_root, subject, subject + '.pos'), 
                   names=['point','x','y','z'], delim_whitespace=True)
pos = df.drop(df.index[0]).to_numpy()

# separate pos into fiducials, electrodes and headshape
# 3 fiducial points at the end
# 1 points for each channel (63 channels)
# the rest are headshape points

pos_fids = pos[-3:,1:] / 100  # change units to m for MNE
pos_elec = pos[-3-len(elec_names):-3,1:] / 100

pos_head = pos[0::100,1:] / 100   # downsample Einscan by 100 for speed

# divide pos by 100 
elec_dict = dict(zip(elec_names,pos_elec))

nas = pos_fids[0,:].astype(float)
lpa = pos_fids[1,:].astype(float)
rpa = pos_fids[2,:].astype(float)
hsp = pos_head.astype(float)

# create head digitisation
digitisation = mne.channels.make_dig_montage(ch_pos=elec_dict, 
                         nasion=nas,
                         lpa=lpa,
                         rpa=rpa,
                         hsp=hsp)

data.set_montage(digitisation, on_missing="ignore")

#%% coregistration

plot_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)

coreg = mne.coreg.Coregistration(data.info, fs_subject, 
                            subjects_dir=subjects_dir)
mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
coreg.fit_fiducials()
coreg.set_grow_hair(0)
coreg.fit_icp(20)
mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)

trans_fname = subject + "_nback_trans.fif"
coreg.trans.save(op.join(deriv_root, subject, trans_fname), overwrite=True)

#%% compute source space

# can change oct5 to other surface source space
src = mne.setup_source_space(
    fs_subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
src.plot(subjects_dir=subjects_dir)

src_fname = subject + "_nback_src.fif"
src.save(op.join(deriv_root, subject, src_fname), overwrite=True)

#%% three layer bem conduction model

conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(
    subject=fs_subject, ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

bem_fname = subject + "_nback_bem.fif"
mne.write_bem_solution(op.join(deriv_root, subject, bem_fname), 
                       bem, overwrite=True)

#%% forward solution

fwd = mne.make_forward_solution(
    data.info,
    trans=coreg.trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True
    )

fwd_fname = subject + "_nback_fwd.fif"
mne.write_forward_solution(op.join(deriv_root, subject, fwd_fname), 
                       fwd, overwrite=True)
