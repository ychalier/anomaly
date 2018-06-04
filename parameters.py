import time

import multiprocessing as mp

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network

from detector import *
from loader import load_detector

import os
import glob


def scorer(estimator, X, y):
    t_start = time.time()
    ypred = estimator.predict(X)
    params = estimator.get_params(deep=True)
    params['estimator'] = type(estimator).__name__
    matrix = metrics.confusion_matrix(y, ypred)
    tpr, fpr, tnr, fnr, ppv, f1 = confusion_ratios(matrix)
    scores = {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'ppv': ppv,
        'f1': f1,
        't': time.time() - t_start
    }
    write(scores, params)
    return f1


def write(scores, params):
    global filename
    pid = os.getpid()

    custom_filename = '{0}.{1}.out.tmp'.format(filename, pid)

    if not os.path.isfile(custom_filename):
        open(custom_filename, 'w').close()

    file = open(custom_filename, 'a')
    file.write("{0}\t{1}\n".format(params, scores))
    file.close()


timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
filename = "params-{0}".format(timestamp)


if __name__ == "__main__":

    detector = load_detector()

    tests = [
        ('MLP', neural_network.MLPClassifier(max_iter=500), {
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'hidden_layer_sizes': [[p] * k for k in range(5) for p in range(1, 50)],
            'alpha': 10.0 ** - np.arange(4, 7),
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam']
        }),
        ('AdaBoost', ensemble.AdaBoostClassifier(), {
            'base_estimator': [tree.DecisionTreeClassifier(max_depth=k)\
                                for k in range(2, 10, 2)],
            'n_estimators': [i for i in range(1, 100, 1)],
            'learning_rate': [x * .1 for x in range(1, 10)]
        })
    ]

    tests_min = [
        ('AdaBoost', ensemble.AdaBoostClassifier(), {
            'n_estimators': [i for i in range(1, 10, 1)],
            'learning_rate': [x * .1 for x in range(1, 2)]
        })
    ]

    for title, base_clf, parameters in tests:
        print(title)
        best_clf, best_params = detector.tune_parameters(base_clf, parameters,\
                                                         verbose=10,\
                                                         n_jobs=mp.cpu_count(),\
                                                         scoring=scorer)
    out_file = open("{0}.out".format(filename), 'w')
    for tmp_out_file in glob.glob('*.tmp'):
        file = open(tmp_out_file, 'r')
        out_file.write(file.read())
        file.close()
        os.remove(tmp_out_file)
    out_file.close()
