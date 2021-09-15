import mne
import getData
import numpy as np
from sklearn.model_selection import train_test_split

#Define the parameters 
subject_Inicial = 1  # use data from subject 1
subject_Final = 7  # use data from subject 1
#runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # use only hand and feet motor imagery runs
runs = [6, 10, 14]
#data = np.empty([2, 2])
#labels = np.empty([2, 2])

#epochs_por_sujeto = [epochs_por_sujeto.append(getData.get_epoch_from_subject_tasks(i, runs, -1, 4)) for i in range(subject_Inicial, subject_Final)]

#for i in range(subject_Inicial, subject_Final):
    #epochs_por_sujeto.append(getData.get_epoch_from_subject_tasks(i, runs, -1, 4))

epochs_por_sujeto = [getData.get_epoch_from_subject_tasks(i, runs, -1, 4) for i in range(subject_Inicial, subject_Final)]

print(len(epochs_por_sujeto))
print(len(epochs_por_sujeto[0]))

data, labels, ids = getData.get_data_labels_ids(epochs_por_sujeto)


print(data)
print(labels)

print(data.shape)
print(labels.shape)

#data = [e.get_data() for e in epochs] #data: array of shape (n_epochs, n_channels, n_times)
#labels = [e.events[:,-1] for e in epochs]

#raw_obj = getData.load_data(subject, runs)
#data, labels = getData.get_data_and_labels(raw_obj, data, labels)

#n_events = len(data[0][:]) # or len(epochs.events)
#print("Number of events: " + str(n_events)) 

#n_channels = len(data[0][0,:]) # or len(epochs.ch_names)
#print("Number of channels: " + str(n_channels))

#n_times = len(data[0][0,0,:]) # or len(epochs.times)
#print("Number of time instances: " + str(n_times))