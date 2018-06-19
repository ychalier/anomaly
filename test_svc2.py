from __future__ import print_function

from detector import *
from loader import load_detector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

print(__doc__)
detector = load_detector()


# Loading the Digits dataset

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

# Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {'multi_class':['ovr','crammer_singer'],'C': [0.01, 0.1, 1, 10, 100, 1000],'tol':[1e-3, 1e-4, 1e-5, 1e-6]
}

#scores = ['precision','f1']
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
    return f1

#for score in scores:
    #print("# Tuning hyper-parameters for %s" % score)
print()
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
scaler.fit(detector.xtrain)

xtrain = scaler.transform(detector.xtrain)
#ytrain = detector.ytrain * 1  # convert bool to 0 or 1

clf = GridSearchCV(LinearSVC(penalty='l1',dual=False), tuned_parameters,cv=5, scoring='f1')
clf.fit(xtrain, detector.ytrain)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f , %0.3f (+/-%0.03f) for %r"
        % (f1,mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = detector.ytest, clf.predict(detector.xtest)
print(classification_report(y_true, y_pred))
print()
