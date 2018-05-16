"""
    A small API for an anomaly detector.
"""

import pandas as pd
import time
import random as rd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn import metrics


def elapsed(tStart):
    return round(time.time() - tStart, 2)


class Range:

    def __init__(self, *args):
        if len(args) == 2:
            self.minx, self.maxx= args[0]
            self.miny, self.maxy= args[1]
        elif len(args) == 4:
            self.minx, self.maxx, self.miny, self.maxy = args


class Detector:

    def __init__(self, df):
        """
            df: pandas DataFrame with 3 cols: class width (mm) and length (mm)
        """
        tStart = time.time()
        self.df = df.dropna(0, 'any')  # remove rows with missing values
        self.df.is_copy = False  # disable SettingWithCopyWarning
        self.clean()
        self.classes = list(set(self.df['class']))
        nRows, nCols = self.df.shape
        assert(nCols == 5)
        print('loaded {0} rows ({1} s)'.format(nRows, elapsed(tStart)))

    def clean(self):
        """
            - adds two new columns: 'duplicate', whether this row appears
              multiple times in the matrix or not, and 'odd', whether the row
              is a 'true' value (by default, yes)
            - removes duplicates rows but keeps the first one each
        """
        self.df.loc[:, 'duplicate'] = self.df.duplicated(keep=False)
        self.df.loc[:, 'odd'] = False
        self.df = self.df.drop_duplicates()

    def format(self, seed=None):
        """
            once odd points were added, re-clean the dataset and split it into
            a training set and a test test, stored in the detecor object.

            seed:
            integer between 0 and 2**32 - 1, used to regenerate a given state
        """
        if seed is None:
            seed = rd.randint(0, 2**32 - 1)
            print('using seed: {0}'.format(seed))

        # shuffling and removing base index as it would break the slicing
        tmp = self.df.sample(frac=1, random_state=seed).reset_index()

        # converting 'class' column from str to int
        tmp['class'] = tmp['class'].map(lambda x: self.classes.index(str(x)))

        # slicing and extracting relevant datasets
        rows = tmp.shape[0] // 2
        self.xtrain = tmp.drop(['odd', 'duplicate', 'index'], axis=1)\
                         .loc[:rows, :]
        self.ytrain = tmp['odd'].loc[:rows]
        self.xtest = tmp.drop(['odd', 'duplicate', 'index'], axis=1)\
                        .loc[rows:, :]
        self.ytest = tmp['odd'].loc[rows:]

    def append_odd_points(self, points):
        """
            points:
            list of width, length and class
        """
        tmp = pd.DataFrame([[p[0], p[1], p[2], False, True] for p in points],
                           columns=self.df.columns)
        self.df = self.df.append(tmp)

    def classify(self, clf, verbose=False):
        tStart = time.time()
        clf.fit(self.xtrain, self.ytrain)
        ypred = clf.predict(self.xtest)
        matrix = metrics.confusion_matrix(self.ytest, ypred)
        if verbose:
            print('computation time: {0}s'.format(elapsed(tStart)))
        return matrix, time.time() - tStart, clf

    def plot(self, rng=None):
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 12))
        plt.title("Plot of input dataset")
        plt.xlabel("width (mm)")
        plt.ylabel("length (mm)")
        colors = cm.rainbow(np.linspace(0, 1, len(self.classes)))
        for class_, color in zip(self.classes, colors):
            # here we use the same variable 'tmp' to avoid allocating too much
            # memory, avoiding MemoryError

            # plotting unique points in triangles
            tmp = self.df.loc[(self.df['class'] == class_) &\
                              (self.df['duplicate'] == False) &\
                              (self.df['odd'] == False)]
            plt.plot(tmp['width'], tmp['length'], '^',
                     label=class_ + " unique", alpha=.5, c=color)

            # plotting duplicate points in circles
            tmp = self.df.loc[(self.df['class'] == class_) &\
                              (self.df['duplicate'] == True) &\
                              (self.df['odd'] == False)]
            plt.plot(tmp['width'], tmp['length'], 'o',
                     label=class_ + " duplicate", alpha=.5, c=color)

            # plotting odd points in squares
            tmp = self.df.loc[(self.df['class'] == class_) &\
                              (self.df['odd'] == True)]
            plt.plot(tmp['width'], tmp['length'], 's',
                     label=class_ + " odd", alpha=.5, c=color)

        if rng is not None:
            plt.xlim(rng.minx, rng.maxx)
            plt.ylim(rng.miny, rng.maxy)

        plt.legend(loc=2)
        plt.show()

    def plot_decision_boundaries(self, clf, title="", step=100, rng=None):
        plt.style.use('seaborn')
        x_min, x_max = min(self.df['width']), max(self.df['width'])
        y_min, y_max = min(self.df['length']), max(self.df['length'])
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_min + x_max) / step),\
                             np.arange(y_min, y_max, (y_min + y_max) / step))
        clf.fit(self.xtrain, self.ytrain)

        plt.figure(figsize=(12, 12))
        plt.title(title)
        colors = cm.rainbow(np.linspace(0, 1, len(self.classes)))
        handles = []
        for class_, color in zip(self.classes, colors):
            classes = [self.classes.index(class_)\
                        for _ in range(len(xx.ravel()))]
            tmp = pd.DataFrame(np.c_[classes, yy.ravel(), xx.ravel()],
                       columns=['class', 'length', 'width'])
            z = clf.predict(tmp)
            z = z.reshape(xx.shape)
            handles.append(mpatches.Patch(color=color, label=class_))
            plt.contour(xx, yy, z, colors=[color])
        plt.legend(handles=handles, loc=2)
        plt.show()


def confusion_ratios(confusion_matrix, verbose=False):
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    tpr = tp / (tp + fn)  # true positives rate
    fpr = fp / (fp + tn)  # false positives rate
    tnr = tn / (tn + fp)  # true negative rate
    fnr = fn / (fn + tp)  # false negative rate
    if verbose:
        print('TPR\tFPR\tTNR\tFNR')
        for x in [tpr, fpr, tnr, fnr]:
            print('{0}\t'.format(round(x, 4)), end='')
        print()
    return tpr, fpr, tnr, fnr
