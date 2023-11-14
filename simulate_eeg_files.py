from scripts.generators import SampleEEG, generate_ica_solution, Stimuli
from scripts.constants import *
import contextlib
import os


# EEG parameters
N_PARTICIPANTS = 10
SAMPLING_RATE = 1000  # in Hz
RESTING_STATE_DURATION = 60  # in seconds
NOISE_IIR_FILTER = [.3, -.3, .04]
N_ICA_COMPONENTS = 32

# mental rotation parameters
N_TRIALS = 196  # trials of mental rotation task
ANGLES = [0, 50, 100, 150]  # unique angles
P_CORRS = [.95, .9, .8, .7]  # probabilities to respond correctly per angle
T_MODES = [1000, 1800, 2600, 3400]  # mode RTs for response in ms (Gumbel distribution) per angle
T_FIG_MIN = 231  # min time for response
T_FIG_MAX = 7500  # max time for response
T_CROSS_MIN = 1000  # min duration for fixation cross
T_CROSS_MAX = 3000  # max duration for fixation cross


def simulate_eeg(subj_id, t):
    signal = SampleEEG()
    signal.set_participant(subj_id)
    signal.simulate_raw(
        sfreq=SAMPLING_RATE,
        duration=t,
        verbose=False
    )
    signal.add_noise(iir_filter=NOISE_IIR_FILTER)
    signal.add_eog()
    return signal


def simulate_taskrelated_eeg(subj_id, f_path):
    stimuli = Stimuli(
        n_trials=N_TRIALS,
        angles=ANGLES, 
        p_corrs=P_CORRS, 
        t_modes=T_MODES, 
        t_fig_min=T_FIG_MIN, 
        t_fig_max=T_FIG_MAX, 
        t_cross_min=T_CROSS_MIN, 
        t_cross_max=T_CROSS_MAX
    )
    stimuli.generate()
    stimuli_duration = stimuli.get_duration_s()
    signal = simulate_eeg(subj_id, int(stimuli_duration * 1.1))
    signal.add_stimuli(stimuli)
    signal.write(f_path, overwrite=True, tmax=stimuli_duration)


def main():
    participant_ids = ''
    for output_dir in [RESTING_STATE_DIR, ICA_DIR, MENTAL_ROTATION_DIR]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for nth_participant in range(N_PARTICIPANTS):
        participant_id = f'id{nth_participant:03d}'
        rs_path = os.path.join(RESTING_STATE_DIR, f'{participant_id}_restingState_raw.fif')
        ica_path = os.path.join(ICA_DIR, f'{participant_id}-ica.fif')
        mr_path = os.path.join(MENTAL_ROTATION_DIR, f'{participant_id}_mentalRotation_raw.fif')

        # resting state simulation
        eeg = simulate_eeg(participant_id, RESTING_STATE_DURATION)
        eeg.write(rs_path, overwrite=True, tmax=RESTING_STATE_DURATION)
        del eeg

        # mental rotation simulation
        simulate_taskrelated_eeg(participant_id, mr_path)
        
        participant_ids += participant_id
        if nth_participant < N_PARTICIPANTS - 1:
            participant_ids += '\n'
        del participant_id, rs_path, ica_path, mr_path        

    # write participant ids
    with open(PARTICIPANT_IDS_FILE, 'w') as f:
        f.write(participant_ids)


if __name__ == '__main__':
    main()
