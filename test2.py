# coding: utf-8

from detector import *
# loading cars
cars = pd.read_csv("CQA_Premium.csv")          [['model_length_mm', 'model_width_mm']]          .rename(columns={
            'model_length_mm': 'length',
            'model_width_mm': 'width'
          })
cars.is_copy = False

cars.loc[:, 'class'] = "car"
print('cars shape:\t{0}'.format(cars.shape))

# loading pedestrians
body = pd.read_csv("body.csv", sep=';')       [['shoulder.girth', 'chest.girth']]       .rename(columns={
           'shoulder.girth': 'width',
           'chest.girth': 'length'
       })

# converting cm to mm
body['width'] = body['width'].map(lambda x: 10 * float(x.replace(',', '.')))
body['length'] = body['length'].map(lambda x: 10 * float(x.replace(',', '.')))

body.is_copy = False
body.loc[:, 'class'] = "human"
print('body shape:\t{0}'.format(body.shape))

moto1 = pd.read_csv("moto.csv", sep=';')        [['Overall length (mm)', 'Overall width (mm)']]        .rename(columns={
            'Overall length (mm)': 'length',
            'Overall width (mm)': 'width'
        }).dropna(0, 'any')

# converting to float
moto1['width'] = moto1['width'].map(lambda x: float(str(x).replace(' ', '')))
moto1['length'] = moto1['length'].map(lambda x: float(str(x).replace(' ', '')))

moto2 = pd.read_csv("moto2.csv", sep=';')        [['LARGHEZZA (mm)', 'LUNGHEZZA (mm)']]        .rename(columns={
            'LARGHEZZA (mm)': 'width',
            'LUNGHEZZA (mm)': 'length'
        })

moto2['width'] = moto2['width'].map(lambda x: float(str(x).replace(' ', '')))
moto2['length'] = moto2['length'].map(lambda x: float(str(x).replace(' ', '')))

# we remove a point at length ~20k that messes up the plot
# during concatenation
moto = moto1.append(moto2[moto2['length'] < 10000])

moto.is_copy = False
moto.loc[:, 'class'] = 'moto'
print('moto shape:\t{0}'.format(moto.shape))

# same here we remove some cars with really odd size
detector = Detector(cars[cars['width'] < 3000].append(body).append(moto))

minx, maxx = 0, max(detector.df['width']) + 1
miny, maxy = 0, max(detector.df['length']) + 1
n = 1000
tmp = []
for k in range(n):
    tmp.append([rd.choice(list(set(detector.df['class']))),
                rd.randint(miny, maxy),
                rd.randint(minx, maxx)])
detector.append_odd_points(tmp)
detector.format()

from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler



parameters={
    'learning_rate': ["invscaling"],
    'hidden_layer_sizes': [x for x in range(1,50)],
    'alpha': 10.0 ** -np.arange(4, 7),
    'activation': ["logistic", "relu"]
}

scaler = StandardScaler()
scaler.fit(detector.xtrain)
#scaler.fit(detector.xtest)


detector.xtrain = scaler.transform(detector.xtrain)
#X_test = scaler.transform(X_test)

#convert bool to 0 or 1
Y_train=detector.ytrain*1


#Y_train=detector.ytrain.transpose()

#Y_train=detector.ytrain.transpose()

mlp = neural_network.MLPClassifier(max_iter=500)
#mlp.fit(detector.xtrain, detector.ytrain)

clf = GridSearchCV(mlp, parameters, verbose=10, n_jobs=75, cv=5)

#clf = GridSearchCV(neural_network.MLPClassifier(),parameters)
clf.fit(detector.xtrain, Y_train)


print(clf.best_estimator_)
print(clf.best_score_)
print(np.argmax(clf.score, axis=None))
#print('Test accuracy:', clf.score(detector.xtest, detector.ytest))

