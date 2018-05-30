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


if __name__ == "__main__":

    detector = load_detector()

    # initialize output file
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = "parameters-search-{0}.out".format(timestamp)
    file = open(filename, 'w')
    file.write("Parameters search from {0}\n\n".format(timestamp))
    file.close()

    tests = [
        ('MLP', neural_network.MLPClassifier(max_iter=500), {
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'hidden_layer_sizes': [[3] * k for k in range(5)],
            'alpha': 10.0 ** - np.arange(4, 7),
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adams']
        }),
        ('AdaBoost', ensemble.AdaBoostClassifier(), {
            'base_estimator': [tree.DecisionTreeClassifier(max_depth=k)\
                                for k in range(2, 10, 2)],
            'n_estimators': [i for i in range(1, 100, 1)],
            'learning_rate': [x * .1 for x in range(1, 10)]
        })
    ]

    for title, base_clf, parameters in tests:
        print(title)
        best_clf, best_params = detector.tune_parameters(base_clf, parameters,\
                                                         verbose=10,\
                                                         n_jobs=mp.cpu_count())
        file = open(filename, 'a')
        file.write("{0}\n{1}\n\n".format(title, str(best_params)\
                        .replace('\n           ', '')))
        file.close()
