import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import numpy as np

def get_epoch_from_subject_tasks(subject, runs, tmin, tmax):
    #Get data and download if needed them to the given path
    files = eegbci.load_data(subject, runs, '/datasets/')
    #For bucle to read raw data from the directory given, each file is a different run
    raws = [read_raw_edf(f, preload=True) for f in files]

    #Limpiamos quitando todas las muestras donde la frequencia de muestreo es 128Hz
    raws = [raw for raw in raws if raw.info['sfreq'] == 160]
    #Combine all loaded runs
    raw_obj = concatenate_raws(raws)

    raw_data = raw_obj.get_data()

    #print(raw_data)
    #print("Number of channels: ", str(len(raw_data)))
    #print("Number of samples: ", str(len(raw_data[0])))

    #plt.plot(raw_data[0,0:4999])
    #plt.title("Raw EEG, electrode 0, samples 0-4999")
    #plt.show()

    #Extract events from raw data
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')
    #print(events)

    #tmin, tmax = -1, 4  # define epochs around events (in s)
    #event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

    epochs = mne.Epochs(raw_obj, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True)

    return epochs

def get_data_labels_ids(epochs):
    data, labels, ids = [], [], []
    
    for epoch in epochs:
        tmp_epoch = epoch[1]
        tmp_labels = tmp_epoch.events[:,-1]
        labels.extend(tmp_labels)
        tmp_id = epoch[0]
        ids.extend([tmp_id]*len(tmp_labels))
        data.extend(tmp_epoch.get_data())
        
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)

    return data, labels, ids

def get_data_labels_ids_nuevo(epochs_por_sujeto):
    data_por_sujeto, labels_por_sujeto, ids_por_sujeto = [], [], []

    for epochs_tarea in epochs_por_sujeto:
        data, labels, ids = [], [], []

        for epoch in epochs_tarea:
            tmp_epoch = epoch[1]
            tmp_labels = tmp_epoch.events[:,-1]
            labels.extend(tmp_labels)
            tmp_id = epoch[0]
            ids.extend([tmp_id]*len(tmp_labels))
            data.extend(tmp_epoch.get_data())
        
        data_por_sujeto = data_por_sujeto.extend(np.array(data))
        labels_por_sujeto = labels_por_sujeto.extend(np.array(labels))
        ids_por_sujeto = ids_por_sujeto.extend(np.array(ids))

    data_por_sujeto = np.array(data)
    labels_por_sujeto = np.array(labels)
    ids_por_sujeto = np.array(ids)

    return data_por_sujeto, labels_por_sujeto, ids_por_sujeto

def get_raw_from_subject_tasks(subject, runs):
    #Get data and download if needed to the given path getting the data for the given subject and runs 
    files = eegbci.load_data(subject, runs, './dataset/')
    # Read each subject and run file and store them into an array
    raws = [read_raw_edf(f, preload=True) for f in files]
    # Clean the data for all the samples where the sfreq is different from 160Hz (128Hz for example)
    raws_filtradas = [raw for raw in raws if raw.info['sfreq'] != 160]
    # Combine the raws from all runs
    if raws_filtradas:
        raw_obj_tasks = concatenate_raws(raws_filtradas)
        # Return the raw object containing the information from all the given runs for the given subject
        return raw_obj_tasks
    else:
        # If there is only raws where the sfreq is different from 160Hz we need to return an empty array
        # in order to prevent an error
        return []

def debuggear(variable):
    print('HOLA')
    return variable