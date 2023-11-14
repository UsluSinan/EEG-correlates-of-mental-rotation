from os import path
import numpy as np

# paths
PROJECT_DIR = path.dirname(path.dirname(path.abspath(__file__)))

DATA_DIR = path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = path.join(PROJECT_DIR, 'output')
TMP_DIR = path.join(PROJECT_DIR, '.tmp')

EPOCHS_DIR = path.join(OUTPUT_DIR, 'epochs')
MODELS_DIR = path.join(OUTPUT_DIR, 'models')
PLOTS_DIR = path.join(OUTPUT_DIR, 'plots')
ICA_DIR = path.join(OUTPUT_DIR, 'ica_solutions')
OUTPUT_DIRS = [
    OUTPUT_DIR, 
    TMP_DIR, 
    EPOCHS_DIR, 
    MODELS_DIR, 
    PLOTS_DIR, 
    ICA_DIR
    ]

RESTING_STATE_DIR = path.join(DATA_DIR, 'restingState')
MENTAL_ROTATION_DIR = path.join(DATA_DIR, 'mentalRotation')
PARTICIPANT_IDS_FILE = path.join(DATA_DIR, 'participant_ids.txt')

SAMPLES_FILE = path.join(OUTPUT_DIR,'samples_after_preprocessing.csv')
RTS_FILE = path.join(OUTPUT_DIR, 'rts.csv')
FEATURES_FILE = path.join(OUTPUT_DIR, 'features_importance.csv')
TUNING_FILE = path.join(OUTPUT_DIR, 'tuning_results.csv')
MODELS_FILE = path.join(OUTPUT_DIR, 'modeling_results.csv')

# EEG
BANDS = {
    'delta': {
        'l_freq': .1, 
        'h_freq': 4,
        'l_trans_bandwidth': .1,
        'h_trans_bandwidth': 2
    },
    'theta': {
        'l_freq': 4, 
        'h_freq': 8,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'alpha': {
        'l_freq': 8, 
        'h_freq': 12,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'lower beta': {
        'l_freq': 12, 
        'h_freq': 16,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'mid beta': {
        'l_freq': 16, 
        'h_freq': 20,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'mid-upper beta': {
        'l_freq': 20, 
        'h_freq': 24,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'upper beta 1': {
        'l_freq': 24, 
        'h_freq': 28,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'upper beta 2': {
        'l_freq': 28, 
        'h_freq': 32,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'lower gamma 1': {
        'l_freq': 32, 
        'h_freq': 36,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    },
    'lower gamma 2': {
        'l_freq': 36, 
        'h_freq': 40,
        'l_trans_bandwidth': 2,
        'h_trans_bandwidth': 2
    }
}
EVENT_TYPE_CHANNEL = 'event_type'
ID_CHANNEL = 'id'
ANGLE_CHANNEL = 'angle'
IS_INVARIANT_CHANNEL = 'is_invariant'
RESPONSE_CHANNEL = 'response'
MIN_MS = 700  # min duration of epochs
MAX_MS = 500  # duration to cut epochs to


# stimuli
NONE = -9
FIXATION_CROSS = 0
OBJECT = 1
RESPONSE = 2

# machine learning
REL_TRAIN_SIZE = .75
LAMBDAS = [np.round(-.9 + (np.power(10, np.log10(10000.9) / 99) ** x), 3) for x in range(100)]
SPOC_COMPONENTS = [0]
COV_METHOD = 'pca'
CV_N_WINDOWS = 3
CV_REL_TRAIN_SIZE = .55
RT_MAD_FACTOR = 2.5