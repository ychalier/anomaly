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

import glob
import os
import sys


def scorer(estimator, X, y):
    """
        A custom scoring function passed as argument to the GridSearchCV.
        It serves two purposes:
         - use f1 as ranking score
         - log all tests (call to function 'write')
    """
    t_start = time.time()
    ypred = estimator.predict(X)
    params = estimator.get_params(deep=True)
    params['estimator'] = type(estimator).__name__  # saving estimator name
    matrix = metrics.confusion_matrix(y, ypred)
    tpr, fpr, tnr, fnr, ppv, f1 = confusion_ratios(matrix)
    scores = {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'ppv': ppv,
        'f1': f1,
        't': time.time() - t_start  # saving how much time it took to predict
    }
    write(scores, params)
    return f1


def write(scores, params):
    """
        Each process spawned by GridSearchCV calls this function when it needs
        to log the test it just performed. It logs them into a dedicated file,
        one for each process. This avoids merge conflicts, as we do not control
        the processing within scikit-learn.
    """
    global filename  # contains the timestamp, same for all processes
    pid = os.getpid()  # process PID is used to differentiates each file
    custom_filename = '{0}.{1}.out.tmp'.format(filename, pid)

    # initializing the file at first
    if not os.path.isfile(custom_filename):
        open(custom_filename, 'w').close()

    # appending tests: params \t scores \n
    file = open(custom_filename, 'a')
    file.write("{0}\t{1}\n".format(params, scores))
    file.close()


timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
filename = "params-{0}".format(timestamp)


if __name__ == "__main__":

    """
        We want to let the possibility to manually interrupt the execution if
        its taking too long, without loosing the data gathered at this point.
        This checks allows user to re-use the script later one just to merge
        the temporary .out.tmp files.
    """
    if not (len(sys.argv) > 1 and sys.argv[1] == "merge"):

        detector = load_detector()

        # add the tests you want to perform here
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
                                    for k in range(2, 10, 1)],
                'n_estimators': [i for i in range(1, 100, 1)],
                'learning_rate': [x * .1 for x in range(1, 10)]
            })
        ]

        # used to debug this script
        tests_minified = [
            ('AdaBoost', ensemble.AdaBoostClassifier(), {
                'n_estimators': [i for i in range(1, 10, 1)],
                'learning_rate': [x * .1 for x in range(1, 2)]
            }),
            ('MLP', neural_network.MLPClassifier(max_iter=500), {
                'hidden_layer_sizes': [[3] * k for k in range(5)],
            })
        ]

        for title, base_clf, parameters in tests:
            print(title)
            result = detector.tune_parameters(base_clf, parameters,\
                                              verbose=10,\
                                              n_jobs=mp.cpu_count(),\
                                              scoring=scorer)
            best_clf, best_params = result
            print('best params:\t', best_params)

    # merge all *.out.tmp files into one .out file
    out_file = open("{0}.out".format(filename), 'w')
    for tmp_out_file in glob.glob('*.tmp'):
        file = open(tmp_out_file, 'r')
        out_file.write(file.read())
        file.close()
        os.remove(tmp_out_file)
    out_file.close()
