# Estimating person-specific neural correlates of mental rotation: A machine learning approach
This is the source code to reproduce the published analyses.

## Content   
    ├── data                    # Raw data files
    ├── output                  # Stores output
    ├── scripts                 # Source files
    ├── main.py                 # Script to execute machine learning
    ├── requirements.txt        # Required python libraries
    ├── restingState_ica.ipynb  # Jupyter notebook to exclude ICA components
    ├── statistical_analysis.R  # Script to execute statistical tests
    ├── LICENSE
    └── README.md

### Data
This folder contains randomly simulated EEG data of 10 participants. Its purpose is to make the code executable and to provide information about the data structure if other data is to be analysed with the scripts.

### Output
This folder stores the output when running the scripts. It also contains the ICA solutions for the simulated resting-state EEG data with randomly excluded components.

### Scripts
All Python wrappers necessary to run the main.py file are stored in this folder.

## Usage
1. Installation
The code has been tested with Python version 3.7.3 and R version 4.3.2 both of which need to be installed. To install the required libraries, run in the main directory:
```sh
python -m pip install -r requirements.txt
```
2. Copy your data into the corresponding data folder (optional)
3. Fit ICA and remove components representing artifacts using the resting-state recording. Run in the main directory:
```sh
jupyter notebook restingState_ica.ipynb
```
4. To process the task-related EEG data, run main.py:
```sh
python main.py
```
5. To run the statistical analysis and generate additional plots, open an R console in the main directory and run:
```sh
source('statistical_analysis.R')
```
