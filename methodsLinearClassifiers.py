from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
import numpy as np

def trainAndPredictGaussianNB(chans, samples, X_train, X_test, y_train, y_test):
    n_components = 2  # pick some components

    #clf = make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), LogisticRegression())
    clf = make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), GaussianNB(priors=None))
    #clf = make_pipeline(GaussianNB(priors=None))

    preds_rg = np.zeros(len(y_test))

    # Reshape back to (trials, channels, samples)
    X_train = X_train.reshape(X_train.shape[0], chans, samples)
    X_test = X_test.reshape(X_test.shape[0], chans, samples)

    # Train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
    # Labels need to be back in single-column format
    clf.fit(X_train, y_train.argmax(axis = -1))

    preds_rg = clf.predict(X_test)

    # Predict with the test set
    y_test = y_test.argmax(axis = -1)
    y_pred = preds_rg

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
