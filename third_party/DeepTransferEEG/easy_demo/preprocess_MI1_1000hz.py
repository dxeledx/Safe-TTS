# this file preprocess raw EEG data into data that are ready for input into feature extraction / neural networks
# preprocessed data will be saved
# download from https://www.bbci.de/competition/iv/ 1000Hz data matlab format
# please correct path down below in the .py file to your own path
import numpy as np
import scipy.io as sio
import os
import mne

dir_path = './BCICIV_1calib_1000Hz_mat'

mne.set_log_level('ERROR')

print('preparing ' + 'MI' + ' 1000Hz data...')
paradigm = 'MI'
num_subjects = 7
sample_rate = 1000
ch_num = 59
# 3 seconds

data = []
labels = []

names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

for i in range(1, num_subjects+1):
    # skip subject 1 and 6, save 5 subjects' data for left/right hand MI EEG classification
    '''
    1 left + foot
    2 left + right
    3 left + right
    4 left + right
    5 left + right
    6 left + foot
    7 left + right
    '''
    if i == 1 or i == 6:
       continue
    mat = sio.loadmat(dir_path + "/BCICIV_calib_ds1" + names[i - 1] +  "_1000Hz.mat")
    X = mat['cnt']
    X = np.transpose(X, (1, 0))
    print(X.shape)
    mrk = mat['mrk']
    pos = mrk['pos'][0, 0][0]

    trials = []
    for start in pos:
        start = int(start)
        trial = X[:, start - 1:start - 1 + 8000]

        # these lines insert channel/electrode information, that are not present in the downloaded data
        # for certain datasets, the EEG data are directly provided WITH such information
        ch_names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CFC7','CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','CCP7','CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5','P3','P1','Pz','P2','P4','P6','PO1','PO2','O1','O2']
        # specify the sampling rate, and electrode type
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=['eeg'] * 59)

        # use MNE package for processing EEG data
        raw = mne.io.RawArray(trial, info)
        data_freqs = []

        trial = raw.get_data()

        print(trial.shape)

        # this line resamples EEG data from 1000Hz to 500 Hz
        trial = mne.filter.resample(trial, down=(1000 / 250))

        # this line band-pass filter EEG data to be within 8-32 Hz, for MI EEG information processing
        trial = mne.filter.filter_data(trial, l_freq=8, h_freq=32, sfreq=250)

        # print('4 seconds')
        trial = trial[:, :1000]
        # print(trial.shape)
        trials.append(trial)
    trials = np.stack(trials)

    y = mrk['y'][0, 0]
    y = (y + 1) / 2
    y = np.array(y).astype(int)
    y = y.reshape(200, )
    print('trials.shape, y.shape', trials.shape, y.shape)  # (59, 300, 200) (200, )
    data.append(trials)
    labels.append(y)
data = np.concatenate(data)
labels = np.concatenate(labels)
print(data.shape, labels.shape)
input('check shape. press enter')
if not os.path.exists('./data/MI1'):
    os.makedirs('./data/MI1')
np.save('./data/MI1/X.npy', data)
np.save('./data/MI1/labels.npy', labels)
print('done')
