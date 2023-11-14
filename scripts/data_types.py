import numpy as np
from functools import partial
from .constants import *
from mne.io import Raw
from os import path
import pickle as pkl

class FrequencyBands(object):
    def __init__(self, signal, bands):
        self.data = {}
        self.names = list(bands.keys())
        for band_name, filter_pars in bands.items():
            band = signal.copy()
            band.filter(**filter_pars)
            self.data[band_name] = band

    def epoch(self, epoch_ranges):
        band_epochs = {}
        for band_name, band in self.data.items():
            epochs = self._epoch(band_name, epoch_ranges)
            band_epochs[band_name] = epochs
        for band_name in self.names:
            self.data[band_name] = band_epochs[band_name]

    def _epoch(self, band_name, epoch_ranges, units='uV'):
        n_epochs = len(epoch_ranges)
        n_channels = len(self.data[self.names[0]].get_channel_types('eeg'))
        n_samples = np.unique([stop - start for start, stop in epoch_ranges])
        assert len(n_samples) == 1, 'epochs are not of identical size'
        epochs = np.zeros((n_epochs, n_channels, int(n_samples)))
        for nth_epoch, (start, stop) in enumerate(epoch_ranges):
            epoch = self.data[band_name].get_data('eeg', start=start, stop=stop, units=units)
            epochs[nth_epoch] = epoch
        return epochs

    def add_info(self, **epoch_info):
        data = {'eeg': self.data}
        data.update(epoch_info)
        self.data = data

    def save(self, f_path):
        with open(f_path, 'wb') as f:
            pkl.dump(self.data, f)
        print(f'{f_path} written')




class Epochs(object):
    def __init__(self, raw):
        self.raw = raw
        self.n = 0
        self._nth_n = -1
        self.range = []
        self.info = []
        self._processing_history = []
        #self.participant = ''
        self._load()

    def _load(self):
        self._load_ranges()
        self._load_info()
        #self.participant = self.raw.info['subject_info']['his_id']
        self.update_n()
        self.archive()

    def _load_ranges(self):
        event_starts = self._get_event_starts()
        for cross_start, object_start, resp_start in event_starts:
            self.range.append([cross_start, object_start])
            self.range.append([object_start, resp_start])

    def _get_event_starts(self):
        event_types = self.get_signal(EVENT_TYPE_CHANNEL)
        n_events = np.sum(event_types != NONE)
        n_trials = np.sum(event_types == FIXATION_CROSS)
        event_starts = np.argwhere(event_types != NONE).reshape(n_trials, n_events // n_trials)
        return event_starts

    def _load_info(self):
        channels = [EVENT_TYPE_CHANNEL, ID_CHANNEL, ANGLE_CHANNEL, IS_INVARIANT_CHANNEL, RESPONSE_CHANNEL]
        signals = [self.get_signal(channel) for channel in channels]
        type_of, id_of, angle_of, invariance_of, response_of = signals
        type_to_str = {
            FIXATION_CROSS: 'fixation_cross',
            OBJECT: 'object',
            NONE: '',
            RESPONSE: 'response'
        }
        for event_idx, (event_start, event_stop) in enumerate(self.range):
            event_info = {
                'type': type_to_str[type_of[event_start]],
                'id': id_of[event_start],
                'angle': angle_of[event_start],
                'is_invariant': invariance_of[event_start],
                'response': response_of[event_stop],
                'duration_ms': self.get_duration_ms(event_idx)
            }
            self.info.append(event_info)

    def get_signal(self, name):
        return self.raw.get_data(name).flatten()

    def update_n(self, **kwargs):
        n = len(self.range)
        self.n = n
        self._nth_n += 1

    def archive(self, **kwargs):
        name = kwargs.get('name', None)
        process_name = self.get_process_name(name)
        value = kwargs.get('value', self.n)
        self._processing_history.append((process_name, str(value)))

    def get_process_name(self, name=None):
        process_name = f'n{self._nth_n}'
        if name is None:
            return process_name
        return process_name + f'_{name}'

    def remove_fixation_crosses(self):
        event_type = self.get_signal(EVENT_TYPE_CHANNEL)
        fixation_crosses = [idx for idx, (event, _) in enumerate(self.range) if event_type[event] == FIXATION_CROSS]
        self.remove(fixation_crosses, name='fixationCross')

    def remove_incorrect_responses(self):
        event_type, invariance, response = [self.get_signal(ch) for ch in [EVENT_TYPE_CHANNEL, IS_INVARIANT_CHANNEL, RESPONSE_CHANNEL]]
        incorrect_resps = []
        for idx, (of_event, to_event) in enumerate(self.range):
            if event_type[of_event] == OBJECT:
                if invariance[of_event] != response[to_event]:
                    incorrect_resps.append(idx)
        self.remove(incorrect_resps, name='incorrectResponses')

    def remove_short_epochs(self, min_ms):
        event_type = self.get_signal(EVENT_TYPE_CHANNEL)
        short_epochs = []
        for idx, (start, stop) in enumerate(self.range):
            dur_ms = self.get_duration_ms(idx)
            if dur_ms < min_ms:
                short_epochs.append(idx)
        self.remove(short_epochs, name=f'lessThan{min_ms}ms')

    def get_duration_ms(self, idx):
        start, stop = self.range[idx]
        dur_ms = (stop - start) / self.raw.info['sfreq'] * 1000
        dur_ms = int(np.round(dur_ms))
        return dur_ms

    def remove(self, indices, name=None):
        for index in reversed(indices):
            del self.range[index], self.info[index]
        self.archive(name=name, value=len(indices))
        self.update_n()
        self.archive()

    def cut(self, to_stop_ms):
        n_samples = int(np.ceil((to_stop_ms / 1000) * self.raw.info['sfreq']))
        for idx, (start, stop) in enumerate(self.range):
            self.range[idx][1] = start + n_samples

    # delete
    def overwrite_processing_history(self, f_path):
        process_names = ','.join(item[0] for item in self._processing_history)
        process_values = ','.join(item[1] for item in self._processing_history)
        header = 'participant,' + process_names + '\n'
        line = self.participant + ',' + process_values + '\n'
        txt = header + line
        with open(f_path, 'w') as f:
            f.write(txt)

    # delete
    def append_processing_history(self, f_path):
        process_values = ','.join(item[1] for item in self._processing_history)
        line = self.participant + ',' + process_values + '\n'
        with open(f_path, 'a') as f:
            f.write(line)

    def save_processing_history(self, f_path, new_file=True, **info):
        if new_file:
            header = list(info.keys()) + [item[0] for item in self._processing_history]
            header = ','.join(header)
            with open(f_path, 'w') as f:
                f.write(header)
        line = list(info.values()) + [item[1] for item in self._processing_history]
        line = '\n' + ','.join(line)
        with open(f_path, 'a') as f:
            f.write(line)

    def apply(self, raw=None):
        if raw is not None:
            self.raw = raw
        n_samples = np.unique([stop - start for start, stop in self.range])
        assert len(n_samples) == 1, 'epochs are not of identical size'
        n_channels = len(self.raw.get_channel_types('eeg'))
        epochs = np.zeros((self.n, n_channels, int(n_samples)))
        for nth_epoch, (start, stop) in enumerate(self.range):
            epochs[nth_epoch] = self._get_eeg_data(start, stop)
        return epochs

    def _get_eeg_data(self, start, stop, units='uV'):
        return self.raw.get_data('eeg', start=start, stop=stop, units=units)
