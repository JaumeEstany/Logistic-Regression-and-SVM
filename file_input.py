import numpy as np
from sklearn.model_selection import KFold
import indexes as ind

indexs = ind.indexs1

db_path = "res/transfusion.data"


# Funcio per a llegir dades en format csv
def load_dataset(path=db_path):
    """
    :param path: path to the file we want to read
    :return:
    """


    data = np.genfromtxt(path, skip_header=1, delimiter=',', dtype=int)
    x = data[:,:4]
    y = data[:,4]

    return x, y


def holdout(x, y, train_ratio):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]

    return x_train, y_train, x_val, y_val


def kFold_indexs(x, splits=4):

    kf = KFold(n_splits=splits, shuffle=True)
    kf.split(x)
    return kf.split(x)


def kFold_dataset(x, y, indexs_t, indexs_v):
    x_train = x[indexs_t]
    x_val = x[indexs_v]
    y_train = y[indexs_t]
    y_val = y[indexs_v]

    return x_train, x_val, y_train, y_val


def LOO_indexs(x, n):

    return kFold_indexs(x, n)


def LOO_dataset(x, y, indexs_t, indexs_v):
    x_train = x[indexs_t]
    x_val = x[indexs_v]
    y_train = y[indexs_t]
    y_val = y[indexs_v]

    return x_train, x_val, y_train, y_val


def standarize(x_train, x_val):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    x_v = x_val - mean[None, :]
    x_v /= std[None, :]
    return x_t, x_v


