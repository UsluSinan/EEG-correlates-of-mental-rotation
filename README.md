# Estimating person-specific neural correlates of mental rotation: A machine learning approach
This is the source code to reproduce the published analyses.

## Content   
    ├── scripts                         # Source files
    ├── estimate_neural_correlates.py   # Script to estimate neural correlates
    ├── requirements.txt                # Required python libraries
    ├── restingState_ica.ipynb          # Jupyter notebook to exclude ICA components
    ├── simulate_eeg_files.py           # Script to simulate EEG data
    ├── statistical_analysis.R          # Script to execute statistical tests
    ├── LICENSE
    └── README.md

### Scripts
All Python wrappers necessary to run the main.py file are stored in this folder.

## Usage
1. Installation
The code has been tested with Python version 3.7.3 and R version 4.3.2 both of which need to be installed. To install the required libraries, run in the main directory:
```sh
python -m pip install -r requirements.txt
```

2. Simulate EEG data by running in the main directory:
```sh
python simulate_eeg_files.py  # simulates EEG data for 10 participants
```

3. Fit ICA and remove components representing artifacts using the resting-state recording. Run in the main directory:
```sh
jupyter notebook restingState_ica.ipynb
```

4. To process the task-related EEG data, run main.py:
```sh
python estimate_neural_correlates.py
```

5. To run the statistical analysis and generate additional plots, open an R console in the main directory and run:
```sh
source('statistical_analysis.R')
```
