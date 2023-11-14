import mne
from mne.io import Raw
from mne.preprocessing import read_ica
from .constants import *
from .data_types import Epochs, FrequencyBands
from .modeling import CrossValidation, EEGmodel, RTmodel, Tuner
from os import path, makedirs
import pickle as pkl
import logging
import shap
import matplotlib.pyplot as plt



def count_epochs(epochs):
    angles = np.array(epochs['angle'])
    n_epochs = {angle: np.sum([angles == angle]) for angle in np.unique(angles)}
    n_epochs['all'] = len(epochs['rt'])
    return n_epochs


def save_aggregated_spoc_patterns(participants):
    # combine all patterns
    evokeds = {band: [] for band in BANDS}
    for nth_participant, participant in enumerate(participants):
        session = Analysis(participant, False)
        model = session.read_models()['eeg']
        info = session.read_mental_rotation_eeg().pick('eeg').info
        info['sfreq'] = 1  # plot 1 component per time unit [s]
        for band in BANDS:
            evoked = mne.EvokedArray(
                model.preprocessor.x_spocs[band].patterns_.T,
                info,
                tmin=0
                )
            evokeds[band].append(evoked)

    # plot grand average across participants per band
    for band in BANDS:
        output_path = path.join(PLOTS_DIR, f'avg-spoc-topomap_{band.replace(" ", "-")}.png')
        evoked_avg = mne.grand_average(evokeds[band])
        p = evoked_avg.plot_topomap(
            time_unit='s', 
            times=SPOC_COMPONENTS, 
            scalings=100, 
            units='a.u.', 
            time_format='#%01d',
            title=f'{band} ({BANDS[band]["l_freq"]}-{BANDS[band]["h_freq"]}Hz)\nSPoC components',
            show=False
            )
        p.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f'{output_path} written')


class Analysis(object):
    def __init__(self, participant, is_first_participant):
        self.participant = participant
        self.is_first_participant = is_first_participant

        # define paths
        self.epochs_path = path.join(EPOCHS_DIR, f'{participant}_epochs.pkl')
        self.models_path = path.join(MODELS_DIR, f'{participant}_models.pkl')
        self.mr_path = path.join(MENTAL_ROTATION_DIR, f'{participant}_mentalRotation_raw.fif')
        self.ica_path = path.join(ICA_DIR, f'{participant}-ica.fif')
        self.plots_dir = path.join(PLOTS_DIR, participant)
        self.shap_plot_path = path.join(self.plots_dir, f'{participant}_shap_summary.png')
        if not path.exists(self.plots_dir):
            makedirs(self.plots_dir)

    def read_mental_rotation_eeg(self):
        signal = Raw(self.mr_path, preload=True, verbose=False)
        return signal

    def apply_ica(self, signal):
        ica = read_ica(self.ica_path, verbose=False)
        signal = ica.apply(signal, verbose=False)
        return signal

    def apply_bandpass_filters(self, signal):
        bands = FrequencyBands(signal, BANDS)
        return bands

    def apply_epoching(self, epoch_ranges, bands):
        # apply epoch ranges to frequency bands
        bands.epoch(epoch_ranges.range)

        # add RTs and angles to EEG epochs
        bands.add_info(
            rt=[epoch['duration_ms'] for epoch in epoch_ranges.info],
            angle=[epoch[ANGLE_CHANNEL] for epoch in epoch_ranges.info]
        )   

        # write processing history (i.e., number of epochs removed) to file
        epoch_ranges.save_processing_history(
            f_path=SAMPLES_FILE,
            new_file=self.is_first_participant,
            participant=self.participant
            )
        return bands

    def save_bands(self, bands):
        bands.save(self.epochs_path)

    @staticmethod
    def split_epochs(epochs):
        # determine absolute train size
        n_epochs = len(list(epochs['eeg'].values())[0])
        n_train = int(np.round(n_epochs * REL_TRAIN_SIZE))  

        # split epochs
        train_epochs = {'eeg': {band_name: band_epochs[:n_train] for band_name, band_epochs in epochs['eeg'].items()}}
        test_epochs = {'eeg': {band_name: band_epochs[n_train:] for band_name, band_epochs in epochs['eeg'].items()}}   

        # add info (e.g., angle, RT) to sets
        for info_name, info_vals in epochs.items():
            if info_name != 'eeg':
                train_epochs[info_name] = info_vals[:n_train]
                test_epochs[info_name] = info_vals[n_train:]    

        return train_epochs, test_epochs

    def save_raw_and_log_rts(self, epochs):
        # write header to file
        if self.is_first_participant:
            with open(RTS_FILE, 'w') as f:
                f.write('participant,unit,angle,rt')
            print(f'{RTS_FILE} written')    

        # extract rts and log(rts) from epochs
        lines = ''
        for unit in ['ms', 'log_ms']:
            rts = {'ms': epochs['rt'], 'log_ms': np.log(epochs['rt'])}[unit]
            for angle, rt in zip(epochs['angle'], rts):
                lines += f'\n{self.participant},{unit},{angle},{rt}' 

        # append rts and log(rts) to file
        with open(RTS_FILE, 'a') as f:
            f.write(lines)
        print(f'{RTS_FILE} appended')

    def tune_lambda(self, epochs):
        logging.getLogger('mne').setLevel(logging.WARNING)
        tuner = Tuner(epochs, CV_N_WINDOWS, CV_REL_TRAIN_SIZE)
        tuned_lambda = tuner.find_lambda(EEGmodel, LAMBDAS)
        tuner.save_tuned_lambda(
            output_path=TUNING_FILE,
            new_file=self.is_first_participant,
            participant=self.participant
            )
        logging.getLogger('mne').setLevel(logging.INFO)
        return tuned_lambda

    @staticmethod
    def train_models(alpha, epochs):
        # get EEG model
        logging.getLogger('mne').setLevel(logging.WARNING)
        eeg_model = EEGmodel(alpha)
        eeg_model.train(
            Xs=epochs['eeg'],
            ys=epochs['rt'],
            angles=epochs['angle']
            )
        logging.getLogger('mne').setLevel(logging.INFO)

        # get RT model
        rt_model = RTmodel()
        rt_model.define_preprocessor(
            ys=epochs['rt'],
            angles=epochs['angle']
            )
        return eeg_model, rt_model

    def interpret_eeg_model(self, eeg_model, train_set, test_set):
        # inspect feature importance
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        eeg_model.plot_feature_importance(
            Xs_train=train_set['eeg'],
            Xs_test=test_set['eeg'],
            output_path=self.shap_plot_path, 
            dpi=600, 
            bbox_inches='tight'
            )

        # save mean(|SHAP|) per feature
        eeg_model.save_shap_values_per_feature(
            Xs_train=train_set['eeg'],
            Xs_test=test_set['eeg'],
            output_path=FEATURES_FILE,
            write_header=self.is_first_participant,
            participant=self.participant
            )
        logging.getLogger('matplotlib').setLevel(logging.INFO)

        # inspect SPoC components
        signal = self.read_mental_rotation_eeg()
        eeg_model.plot_spoc_components(
            output_dir=self.plots_dir,
            info=signal.pick('eeg').info,
            _id=self.participant,
            dpi=600
            )

    def save_models(self, eeg_model, rt_model):
        models = {'eeg': eeg_model, 'rt': rt_model}
        with open(self.models_path, 'wb') as f:
            pkl.dump(models, f)
        print(f'{self.models_path} written')

    def read_models(self):
        with open(self.models_path, 'rb') as f:
            return pkl.load(f)

    def read_epochs(self):
        with open(self.epochs_path, 'rb') as f:
            return pkl.load(f)

    def evaluate_intraindividual_performance(self):
        models = self.read_models()
        epochs = self.read_epochs()

        # select test set
        train_set, test_set = self.split_epochs(epochs)
        del epochs  

        # count samples per angle
        sets = {'train': train_set, 'test': test_set}
        set_sizes = {set_name: count_epochs(_set) for set_name, _set in sets.items()}
        del sets, train_set 

        # prepare output file
        if self.is_first_participant:
            with open(MODELS_FILE, 'w') as f:
                header = 'trainParticipant,testParticipant,model,testSetPreprocessor,angularDisparity,n_trainSamples,n_testSamples,measure,value'
                f.write(header)
            print(f'{MODELS_FILE} written') 

        # for EEG and RT model
        for model_name, model in models.items():    

            # evaluate based on scaled and raw labels
            for measure, evaluate_model in zip(['MAE', 'MAE_raw'], [model.evaluate, model.evaluate_raw]):   

                # evaluate model
                maes = evaluate_model(
                Xs=test_set['eeg'],
                ys=test_set['rt'],
                angles=test_set['angle']
                )       

                # save evaluation
                lines = ''
                for angle, mae in maes.items():
                    line = f'\n{self.participant},same_as_trainParticipant,{model_name},all_from_trainParticipant,{angle},{set_sizes["train"][angle]},{set_sizes["test"][angle]},{measure},{mae}'
                    lines += line
                with open(MODELS_FILE, 'a') as f:
                    f.write(lines)
        print(f'{MODELS_FILE} appended with intraindividual evaluation')

    def evaluate_interindividual_performance(self, other_participants):

        # load epochs of trainParticipant
        train_participants_epochs = self.read_epochs()

        # count train sizes
        train_participants_train_set, _ = self.split_epochs(train_participants_epochs)
        set_sizes = {'train': count_epochs(train_participants_train_set)}   

        # average test sizes of all other participants
        test_set_sizes = {angle: [] for angle in np.unique(train_participants_train_set['angle'])}
        test_set_sizes['all'] = []
        set_sizes['test'] = test_set_sizes  

        # count test sizes of all other participants
        for other_participant in other_participants:
            other_session = Analysis(other_participant, False)
            test_participants_epochs = other_session.read_epochs()
            _, test_participants_test_set = other_session.split_epochs(test_participants_epochs)
            test_set_size = count_epochs(test_participants_test_set)    

            # count test sizes for each angle
            for angle, set_size in test_set_size.items():
                set_sizes['test'][angle].append(set_size)   

        # average test sizes across other participants
        for angle, sizes in set_sizes['test'].items():
            set_sizes['test'][angle] = np.mean(sizes)   

        # evaluate models with different preprocessors
        preprocessors = [
        'all_from_trainParticipant', 
        'rt_from_testParticipant', 
        'SPoC_from_testParticipant', 
        'all_from_testParticipant'
        ]
        maes_to_average = {}
        for preprocessor in preprocessors:
            maes_to_average[preprocessor] = {angle: [] for angle in np.unique(train_participants_train_set['angle'])}
            maes_to_average[preprocessor]['all'] = []  

        # for every other participant
        for other_participant in other_participants:    
            other_session = Analysis(other_participant, False)

            # evaluate with different preprocessors
            for preprocessor in preprocessors:
                maes = self.evaluate_interindividual_performance_with_preprocessor(
                    preprocessor=preprocessor,
                    other_participants_session=other_session
                    )
                for angle, mae in maes.items():
                    maes_to_average[preprocessor][angle].append(mae)    

        # average MAEs across testParticipants and save evaluation
        lines = ''
        for preprocessor in maes_to_average.keys():
            for angle, maes in maes_to_average[preprocessor].items():   

                # average MAEs per preprocessor and angle
                mae = np.mean(maes) 

                # prepare line to write
                line = f'\n{self.participant},other_participant,eeg,{preprocessor},{angle},{set_sizes["train"][angle]},{set_sizes["test"][angle]},MAE,{mae}'
                lines += line   

        # save evaluation
        with open(MODELS_FILE, 'a') as f:
            f.write(lines)  

        print(f'{MODELS_FILE} appended with interindividual evaluation')

    def evaluate_interindividual_performance_with_preprocessor(self, preprocessor, other_participants_session):
        with_rt_from_test_participant, with_spoc_from_test_participant = {
        'all_from_trainParticipant': [False, False],
        'rt_from_testParticipant': [True, False],
        'SPoC_from_testParticipant': [False, True],
        'all_from_testParticipant': [True, True]
        }[preprocessor] 

        # load models
        train_participants_model = self.read_models()['eeg']
        test_participants_model = other_participants_session.read_models()['eeg'] 

        # load test set
        epochs = other_participants_session.read_epochs()
        _, test_participants_test_set = other_participants_session.split_epochs(epochs)    

        if with_rt_from_test_participant:
            train_participants_model.preprocessor.y_means = test_participants_model.preprocessor.y_means
            train_participants_model.preprocessor.y_sds = test_participants_model.preprocessor.y_sds
        if with_spoc_from_test_participant:
            train_participants_model.preprocessor.x_spocs = test_participants_model.preprocessor.x_spocs    

        maes = train_participants_model.evaluate(
                Xs=test_participants_test_set['eeg'],
                ys=test_participants_test_set['rt'],
                angles=test_participants_test_set['angle']
            )
        return maes
