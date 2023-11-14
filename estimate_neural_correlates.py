import os
from scripts.constants import PARTICIPANT_IDS_FILE, OUTPUT_DIRS, MIN_MS, MAX_MS
from scripts.analysis_steps import Analysis, save_aggregated_spoc_patterns
from scripts.data_types import Epochs


def prepare_output_dirs():
    for output_dir in OUTPUT_DIRS:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


def list_participants():
    with open(PARTICIPANT_IDS_FILE, 'r') as f:
        participants = f.read().splitlines()
    return participants


def prepare_epochs(session):
    eeg = session.read_mental_rotation_eeg()
    eeg = session.apply_ica(eeg)
    bands = session.apply_bandpass_filters(eeg)
    epoch_ranges = get_epoch_ranges(eeg)
    bands = session.apply_epoching(epoch_ranges, bands)
    session.save_bands(bands)
    return bands.data


def get_epoch_ranges(signal):
    epoch_ranges = Epochs(signal)
    epoch_ranges.remove_fixation_crosses()
    epoch_ranges.remove_incorrect_responses()
    epoch_ranges.remove_short_epochs(MIN_MS)
    epoch_ranges.cut(MAX_MS)
    return epoch_ranges


def main():
    prepare_output_dirs()
    participants = list_participants()

    # save EEG and RT model per participant
    for nth_participant, participant in enumerate(participants):

        # initiate analysis session
        session = Analysis(participant=participant, is_first_participant=nth_participant==0)

        # prepare epochs
        epochs = prepare_epochs(session)

        # split epochs
        train_set, test_set = session.split_epochs(epochs)
        del epochs

        # save unscaled and log-transformed RTs
        session.save_raw_and_log_rts(train_set)

        # tune lambda for EEG model
        tuned_lambda = session.tune_lambda(train_set)

        # train EEG model with tuned lambda and define RT model
        eeg_model, rt_model = session.train_models(tuned_lambda, train_set)
        del tuned_lambda

        # interpret EEG model
        session.interpret_eeg_model(eeg_model, train_set, test_set)
        del train_set, test_set

        # save both models
        session.save_models(eeg_model, rt_model)
        del eeg_model, rt_model

    # plot averaged topographic activity per feature
    save_aggregated_spoc_patterns(participants)


    # evaluate EEG and RT models
    for nth_participant, participant in enumerate(participants):

        # initiate analysis session
        session = Analysis(participant=participant, is_first_participant=nth_participant==0)

        # EEG vs RT model within the same participant
        session.evaluate_intraindividual_performance()

        # assess the generalizability of the EEG model
        other_participants = [p for p in participants if p != participant]
        session.evaluate_interindividual_performance(other_participants)
        del other_participants


if __name__ == '__main__':
    main()
