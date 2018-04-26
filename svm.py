from sklearn import svm


def linear_svm(x_train, y_train, x_val, y_val, C=0.01, gamma=0.001, probability=True):
    svclin = svm.SVC(C=C, kernel='linear', gamma=gamma, probability=probability)
    svclin.fit(x_train, y_train)
    y_pred = svclin.predict(x_val)

    return svclin, y_pred


def poly_svm(x_train, y_train, x_val, y_val, degree=2, C=0.01, gamma=0.01, probability=True, tol=0.01):
    svcpol = svm.SVC(C=C, kernel="poly", gamma=gamma, degree=degree, probability=probability, tol=tol, cache_size=7000)
    svcpol.fit(x_train, y_train)
    y_pred = svcpol.predict(x_val)

    return svcpol, y_pred


def rbf_svm(x_train, y_train, x_val, y_val, C=0.01, gamma=0.001, probability=True, tol=0.001):
    svcrbf = svm.SVC(C=C, kernel="rbf", gamma=gamma, probability=probability, tol=tol, cache_size=7000)
    svcrbf.fit(x_train, y_train)
    y_pred = svcrbf.predict(x_val)

    return svcrbf, y_pred
