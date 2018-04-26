import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import file_input as fi
from sklearn.linear_model import LogisticRegression
import sklearn.decomposition.pca as pca



def reduce_dimensionsInto1(x_train, y_train, x_val):

    pcaObj = pca.PCA(1)

    newDimension = pcaObj.fit_transform(x_train, y_train)

    indexes = newDimension[:, 0].argsort()
    newDimension = newDimension[indexes, :]
    y_train = y_train[indexes]

    x_val = pcaObj.transform(x_val)

    return newDimension, y_train, x_val


def reduce_dimensionsInto2(x_train, y_train, x_val):

    pcaObj = pca.PCA(2)

    newDimensions = pcaObj.fit_transform(x_train, y_train)

    indexes = newDimensions[:, 0].argsort()
    newDimensions = newDimensions[indexes, :]
    y_trainOrdered = y_train[indexes]

    x_val = pcaObj.transform(x_val)

    return newDimensions, y_trainOrdered, x_val


def represent_logistic_regression(x_val, y_pred, model):
    probs = model.predict_proba(x_val)

    indexs = probs[:, 1].argsort()
    probs = probs[indexs]
    x_val = x_val[indexs]
    y_pred = y_pred[indexs]

    plt.figure()
    plt.scatter(x_val, y_pred == 1, color="red")
    plt.scatter(x_val, y_pred == 0, color="green")
    plt.plot(x_val, probs[:, 1])
    plt.show()


def make_meshgrid(x, y, h=None):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """


    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1


    if h is None:
        h_x = (x.max() - x.min())/400
        h_y = (y.max() - y.min())/400
        h = h_y

        if h_x > h_y:
            h = h_x

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def represent_clfs(x, y):
    #X = x[:, :2]


    # import some data to play with
    #x, y = fi.load_dataset()

    indexs = indices = np.arange(x.shape[0])
    np.random.shuffle(indexs)
    n_samples = int(round(0.2 * x.shape[0]))

    indexs = indexs[:n_samples]
    X = x[indexs, :2]
    y = y[indexs]

    pcaObj = pca.PCA(2)
    pcaObj.fit(X)
    X = pcaObj.transform(X)

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter

    models = (LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001),
              svm.SVC(kernel='linear', C=C, random_state=0),
              svm.SVC(kernel='rbf', gamma=0.07, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('Logistic regression',
              'SVC with lineal kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    sub = np.append(sub[0], sub[1])

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, None)

    i = 0

    tuple = zip(models, titles, sub)

    for clf, title, ax in tuple:
        print i
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()
    plt.savefig("results/Representation/classificators.png")

