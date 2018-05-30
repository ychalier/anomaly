import pandas as pd
from detector import *

def load_detector():
    # loading cars
    cars = pd.read_csv("CQA_Premium.csv")\
              [['model_length_mm', 'model_width_mm']]\
              .rename(columns={
                'model_length_mm': 'length',
                'model_width_mm': 'width'
              })
    cars.is_copy = False

    cars.loc[:, 'class'] = "car"

    # loading pedestrians
    body = pd.read_csv("body.csv", sep=';')\
           [['shoulder.girth', 'chest.girth']]\
           .rename(columns={
               'shoulder.girth': 'width',
               'chest.girth': 'length'
           })

    # converting cm to mm
    body['width'] = body['width'].map(lambda x: 10 * float(x.replace(',', '.')))
    body['length'] = body['length'].map(lambda x: 10 * float(x.replace(',', '.')))

    body.is_copy = False
    body.loc[:, 'class'] = "human"

    moto1 = pd.read_csv("moto.csv", sep=';')\
            [['Overall length (mm)', 'Overall width (mm)']]\
            .rename(columns={
                'Overall length (mm)': 'length',
                'Overall width (mm)': 'width'
            }).dropna(0, 'any')

    # converting to float
    moto1['width'] = moto1['width'].map(lambda x: float(str(x).replace(' ', '')))
    moto1['length'] = moto1['length'].map(lambda x: float(str(x).replace(' ', '')))

    moto2 = pd.read_csv("moto2.csv", sep=';')\
            [['LARGHEZZA (mm)', 'LUNGHEZZA (mm)']]\
            .rename(columns={
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
    return detector
