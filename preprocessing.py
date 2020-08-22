import pandas as pd
import mne 
from mne.channels import read_custom_montage
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import csv



def load_data(filename, method):
    data = pd.read_csv(filename) 
    data = data.fillna(-1)
    data = data.drop(["EOG", "Time"], axis = 1)
    # 4: no reaction
    # 5: reaction
    data['FeedBackEvent'].replace({1:(5)},inplace=True)
    data.iloc[ :, 0: len(data.T) - 1] /= 1e6

    if method == "train":
        d = get_label_dic()
        idx = 1
        for i in range((len(data["FeedBackEvent"]))):
            if data["FeedBackEvent"][i] == 5:
                session = "0" + str(idx).zfill(2) if idx < 100 else str(idx)
                data.iloc[i: i+1 , 56: 57] =  int(int (d[filename[18:28] + "_FB" + session] ) + 1)
                idx += 1
    return data


def mneLoadIn(new_data, method):
    
    l=len(new_data.T)
    ch_names = list(new_data.T.index)  
    sfreq = 200
    ch_types=['eeg' for i in range(l-1)]
    ch_types.append('stim')
    info = mne.create_info(ch_names, sfreq, ch_types)# , montage=montage
    raw = mne.io.RawArray(new_data.T.values, info,copy='info')
    print('Plot raw EEG data!')
    print('Downsampling... ...')
    print('Original sfreq:',raw.info['sfreq'])
    raw=raw.resample(128, npad='auto')
    print('sfreq after downsampling:',raw.info['sfreq'])
             # Re-reference
    print('Re-reference... ...')
    raw.set_eeg_reference()# ref_channels='average',ch_type='eeg'
    #         Centering
    print('Centering... ...')
    for ch in range(l-1): # l-1: last channel is stim channel
        raw._data[ch,:]=raw._data[ch,:]-np.mean(raw._data[ch,:])

    
            # Bandpass filtering [1,40] Hz
    print('Bandpass filtering... ...')
    h_freq = 40
    raw=raw.filter(1, h_freq, method='fir',picks=np.arange(l-1))# ,iir_params=iir_params
    print('Plot EEG data after bandpass [1,{}] Hz!'.format(h_freq))
    # raw.plot()
            # Find events in EEG data
    events = mne.find_events(raw)
    # print(events)
    # raw.plot(events=events)
    baseline=(0, 1.25)
    event = [5] if method == "test" else [1,2]
    epochs = mne.Epochs(raw, events, tmin= 0, tmax = 1.25, event_id = event, baseline=baseline,
                        picks=np.arange(l-1),proj='delayed')#, flat=flat_criteria
    # print(epochs)
    print('Plot epochs!')
    return epochs

def get_label_dic():
    reader = csv.reader(open('kaggle/TrainLabels.csv', 'r'))
    d = {}
    for row in reader:
       k, v = row
       d[k] = v
    # print(d)
    return d


train_set = ["02", "06", "07", "11", "12", "13", "14", "16", "17", "18", "20", "21", "22", "23", "24", "26"]
test_set = ["01", "03", "04", "05", "08", "09", "10", "15", "19", "25"]
kernel, ch , sample = 1, 56, 161


# for subject in train_set:
#     x_train = []
#     y_train = []
#     for session in range(1,6):
#         label = []
#         filename = "Data_S" + subject + "_Sess0" + str(session)
#         data = load_data("kaggle/train/" + filename + ".csv", "train")
#         epochs = mneLoadIn(data, "train")
#         for event in epochs.events:
#             # print(event[2])
#             y_train.append(event[2])
#         # x_train.append(list(epochs.get_data()))
#         if session == 1:
#             x_train = epochs.get_data()
#             # print(x_train.shape)

#         else:
#             x_train = np.append(x_train, epochs.get_data())
#     x_train = np.reshape(x_train, ( int(x_train.shape[0]/ch/sample), kernel , ch, sample))
#     y_train = np.array(y_train)
#     np.savez_compressed('bci_challenge_training_npz/' + filename[:-2] +'.npz', x_train = x_train, y_train = y_train)

for subject in test_set:
    x_train = []
    for session in range(1,6):
        filename = "Data_S" + subject + "_Sess0" + str(session)
        data = load_data("kaggle/test/" + filename + ".csv", "test")
        epochs = mneLoadIn(data, "test")
        if session == 1:
            x_train = epochs.get_data()
            # print(x_train.shape)
        else:
            x_train = np.append(x_train, epochs.get_data())
    x_train = np.reshape(x_train, ( int(x_train.shape[0]/ch/sample), kernel , ch, sample))
    # print(x_train.shape)

    np.savez_compressed('bci_challenge_testing_npz/' + filename[:-2] +'.npz', x_train = x_train)
