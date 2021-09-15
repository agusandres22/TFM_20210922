import numpy as np
from tensorflow.keras import utils as np_utils
from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix

def transformParameters(kernels, chans, samples, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test):
    # Transform the label into categorical variables
    y_train = np_utils.to_categorical(y_train-1)
    y_validate = np_utils.to_categorical(y_validate-1)
    y_test = np_utils.to_categorical(y_test-1)

    # Transform the data to NHWC format (trials, channels, samples, kernels)
    # In this case we have the number of channels set to 64 and also we have 809 time points. 
    # We also set the kernel value to 1.
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def trainModel_EEG(kernels, chans, samples, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test, epochs_param):
    # Configure the EEGNet-8,2,16 model with a kernel length of chans samples
    model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, dropoutType = 'Dropout')

    # Compile the model and choose the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    # Count the number of parameter of the model
    numParams = model.count_params()
    print(numParams)

    # Set the path where the model checkpoints will be stores
    checkpointer = ModelCheckpoint(filepath='models/checkpoint.h5', verbose=0, save_best_only=True)

    # Define each class weights
    class_weights = {0:1, 1:1, 2:1}

    # Train the model
    fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = epochs_param, verbose = 0, validation_data=(X_validate, y_validate), callbacks=[checkpointer], class_weight = class_weights)

    # Save the trained model
    model.save('models/modeloEntrenado_EEG.h5')
    
    # Save the model training history
    np.save('models/history1.npy', fittedModel.history)

    return model, fittedModel

def predictModel_EEG(model, X_test, y_test):
    # Predict with the test set
    y_test = y_test.argmax(axis = -1)
    probs = model.predict(X_test)
    y_pred = probs.argmax(axis = -1)

    #importing confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Rest', 'Close left fist', 'Close right fist']))