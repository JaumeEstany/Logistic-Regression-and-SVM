import numpy as np
from sklearn.linear_model import LogisticRegression


def logisticRegression(x_t, y_t, x_v, y_v):

    #Creacio i entrenament del regressor logistic amb conjunt d'entrenament
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    logireg.fit(x_t, y_t)

    #Classificacio de conjunt de validacio
    y_pred = logireg.predict(x_v)

    return logireg, y_pred


def logisticRegressionLOO(x_t, y_t, x_v, y_v):

    #Creem el regresor logistic
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    # l'entrenem
    logireg.fit(x_t, y_t)

    # classifiquem les mostres de validacio
    y_pred = logireg.predict(x_v)
    # calculem l'error
    if (y_v == y_pred):
        return 1.0
    else:
        return 0.0

