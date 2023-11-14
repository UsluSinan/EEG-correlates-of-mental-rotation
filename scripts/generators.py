from functools import partial
import numpy as np
import pickle as pkl
import os

import mne
from mne.datasets import sample
from mne.simulation import simulate_raw, simulate_sparse_stc, add_noise, add_ecg, add_eog


EEG_CHANNELS = {
    'EEG 001': 'Fp1',
    'EEG 003': 'Fp2',
    
    'EEG 008': 'F7',
    'EEG 010': 'F3',
    'EEG 012': 'Fz',
    'EEG 014': 'F4',
    'EEG 016': 'F8',
    
    'EEG 018': 'FC5',
    'EEG 020': 'FC1',
    'EEG 021': 'FC2',
    'EEG 023': 'FC6',
    
    'EEG 026': 'T7',
    'EEG 028': 'C3',
    'EEG 030': 'Cz',
    'EEG 032': 'C4',
    'EEG 034': 'T8',
    
    'EEG 036': 'TP8',
    'EEG 037': 'CP5',
    'EEG 039': 'CP1',
    'EEG 040': 'CP2',
    'EEG 042': 'CP6',
    'EEG 043': 'TP10',
    
    'EEG 044': 'P7',
    'EEG 046': 'P3',
    'EEG 048': 'Pz',
    'EEG 050': 'P4',
    'EEG 052': 'P8',
    
    'EEG 054': 'PO9',
    'EEG 057': 'O1',
    'EEG 058': 'Oz',
    'EEG 059': 'O2',
    'EEG 056': 'PO10',
}


def generate_ica_solution(eeg, f_path, verbose=True):
    n_components = len(eeg.raw.ch_names)
    ica = {
        'pre_whitener': np.array([[np.random.rand() * 1e-5]] * n_components),
        'pca_mean': np.random.normal(.01, .02, (n_components, 1)),
        'proj_mat': np.random.normal(.01, .2, (n_components, n_components))
    }
    with open(f_path, 'wb') as f:
        pkl.dump(ica, f)
    if verbose:
        print(f'{f_path} written')


def data_fun(times, amp_min=1.0e-9, amp_max=2.5e-8, n_amps=20):
    amps = np.linspace(amp_min, amp_max, n_amps)
    amp = np.random.choice(amps, 1)
    return amp * np.sin(20 * np.pi * times)


def interleave(*arrays):
    return np.dstack(arrays).reshape(len(arrays[0]), -1).flatten()


class Stimuli(object):
    def __init__(
        self, 
        n_trials, 
        angles, 
        p_corrs, 
        t_modes, 
        t_fig_min, 
        t_fig_max, 
        t_cross_min, 
        t_cross_max,
        _missing=-9
    ):
        self.n_trials = n_trials  # trial: fixation cross -> mental rotation -> response
        self.angles = angles
        self.p_corrs = p_corrs
        self.t_modes = t_modes
        self.t_fig_min = t_fig_min
        self.t_fig_max = t_fig_max
        self.t_cross_min = t_cross_min
        self.t_cross_max = t_cross_max
        self._none = -9
        self._missing = -1
        self.items = []
        self.info = ['event_type', 'id', 'angle', 'is_invariant', 'response', 'duration_ms']
        
    def generate(self):
        event_types = [0, 1, 2] * self.n_trials
        ids = self.generate_ids()
        angles = self.generate_angles()
        invariances = self.generate_invariances()
        responses, durations_ms = self.generate_responses_and_durations(angles, invariances)
        stims = [list(stim) for stim in zip(event_types, ids, angles, invariances, responses, durations_ms)]
        self.items = stims
        
    def generate_ids(self):
        cross_ids = list(range(self.n_trials))
        fig_ids = np.random.choice(range(self.n_trials, self.n_trials * 3), self.n_trials, replace=False)
        resp_ids = self.generate_none()
        stim_ids = interleave(cross_ids, fig_ids, resp_ids)
        return stim_ids
    
    def generate_angles(self):
        cross_angles = self.generate_none()
        fig_angles = self.angles * int(self.n_trials / len(self.angles))
        np.random.shuffle(fig_angles)
        resp_angles = self.generate_none()
        stim_angles = interleave(cross_angles, fig_angles, resp_angles)
        return stim_angles
    
    def generate_invariances(self):
        cross_invariances = self.generate_none()
        fig_invariances = [0, 1] * int(self.n_trials / 2)
        np.random.shuffle(fig_invariances)
        resp_invariances = self.generate_none()
        stim_invariances = interleave(cross_invariances, fig_invariances, resp_invariances)
        return stim_invariances
    
    def generate_responses_and_durations(self, angles, invariances):
        cross_responses = self.generate_none()
        fig_responses = self.generate_none()

        t_cross_diff = self.t_cross_max - self.t_cross_min
        cross_durations = np.array(t_cross_diff * np.random.sample(self.n_trials) + self.t_cross_min).astype(int)
        resp_durations = [1] * self.n_trials

        resps = []
        fig_durations = []
        for fig_angle, fig_invariance in zip(angles[1::3], invariances[1::3]):
            
            mode = self.t_modes[self.angles.index(fig_angle)]
            fig_duration = int(np.random.gumbel(mode, mode / 4))
            fig_duration = self.t_fig_min if fig_duration < self.t_fig_min else fig_duration
            
            p_corr = self.p_corrs[self.angles.index(fig_angle)]
            resp_is_corr = np.random.choice(2, 1, p=[1 - p_corr, p_corr])
            resp = fig_invariance if resp_is_corr else 1 - fig_invariance
            resp = self._missing if fig_duration > self.t_fig_max else resp
            resps.append(resp)
            
            fig_duration = self.t_fig_max if fig_duration > self.t_fig_max else fig_duration
            fig_durations.append(fig_duration)
            
        responses = interleave(cross_responses, fig_responses, resps)
        durations = interleave(cross_durations, fig_durations, resp_durations)
        return responses, durations            
        
    def generate_none(self):
        return [self._none] * self.n_trials
    
    def get_duration_ms(self):
        durations_ms = [item[-1] for item in self.items]
        duration_ms = sum(durations_ms)
        return duration_ms
    
    def get_duration_s(self):
        duration_ms = self.get_duration_ms()
        duration_s = np.ceil(duration_ms / 1000).astype(int)
        return duration_s
    
    def get_items(self):
        return self.items
    
    def as_signal(self, sfreq): #['event_type', 'id', 'angle', 'is_invariant', 'response', 'duration_ms']
        signal = np.ones((self.get_duration_ms(), len(self.info[:-1]))) * self._none
        row = 0
        for item in self.items:
            signal[row] = item[:-1] # add response later + ch for event type (fix, stim, resp)
            row += item[-1]
        return signal


class SampleEEG(object):
    def __init__(self):
        self.dir = os.path.join(sample.data_path(), 'MEG', 'sample')
        self.raw = None
        self.fwd = None
        self.sfreq = None
        self.duration = None
        self.load_raw()
        self.load_forward_solution()

    def load_raw(self, f_name='sample_audvis_raw.fif'):
        f_path = os.path.join(self.dir, f_name)
        raw = mne.io.read_raw_fif(f_path)
        self.sfreq = raw.info['sfreq']
        self.raw = raw

    def load_forward_solution(self, f_name='sample_audvis-meg-eeg-oct-6-fwd.fif'):
        f_path = os.path.join(self.dir, f_name)
        fwd = mne.read_forward_solution(f_path)
        self.fwd = fwd

    def simulate_raw(self, sfreq, verbose, **kwargs):
        stcs = self.simulate_source_time_courses(**kwargs)
        raws = [mne.simulation.simulate_raw(self.raw.info, [stc], forward=self.fwd, verbose=verbose) for stc in stcs]
        raw = mne.concatenate_raws(raws)
        raw = raw.pick_channels(list(EEG_CHANNELS)).copy()
        raw.rename_channels(EEG_CHANNELS)
        raw.resample(sfreq=sfreq)
        raw.annotations.delete(range(len(raw.annotations.description)))
        self.sfreq = sfreq
        self.raw = raw

    def add_noise(self, seed=None, **kwargs):
        self.raw.add_reference_channels('FCz')
        self.raw.set_eeg_reference(ref_channels='average')
        cov = mne.make_ad_hoc_cov(self.raw.info)
        mne.simulation.add_noise(self.raw, cov, random_state=seed, **kwargs)
        self.raw.set_eeg_reference(ref_channels=['FCz'])
        self.raw = self.raw.pick_channels(self.raw.ch_names[:-1])

    def add_eog(self, seed=None):
        mne.simulation.add_eog(self.raw, random_state=seed)
        
    def add_stimuli(self, stimuli):
        info = mne.create_info(
            stimuli.info[:-1],
            self.sfreq,
            ['stim'] * len(stimuli.info[:-1])
        )
        data = stimuli.as_signal(self.sfreq).T
        data_padded = np.ones((data.shape[0], self.raw.get_data().shape[1])) * stimuli._none
        data_padded[:, :data.shape[1]] = data
        stimuli_signal = mne.io.RawArray(data_padded, info)
        for key in self.raw.info.keys():
            if key not in ['chs', 'ch_names', 'nchan']:
                stimuli_signal.info[key] = self.raw.info[key]
        self.raw.add_channels([stimuli_signal])
        
    def simulate_source_time_courses(self, n_dipoles=16, duration=60, epoch_duration=2, seed=None):
        self.duration = duration
        times = np.arange(self.sfreq * epoch_duration) / self.sfreq
        n_epochs = int(np.ceil(duration / epoch_duration))
        stcs = []
        for epoch in range(n_epochs):
            stc = mne.simulation.simulate_sparse_stc(
                src=self.fwd['src'],
                n_dipoles=n_dipoles,
                times=times,
                data_fun=data_fun,
                random_state=seed
                )
            stcs.append(stc)
        return stcs

    def set_participant(self, subj_id):
        self.raw.info['subject_info'] = {'his_id': subj_id}

    def write(self, f_path, **kwargs):
        self.raw.save(f_path, **kwargs)
        print(f'{f_path} written')
