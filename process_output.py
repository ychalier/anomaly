def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# if it does not work, be sure to use command:
#     export MPLBACKEND="agg"
matplotlib.use('Agg')

import pandas as pd
import time
import sys

def count_layers(line):
    if len(line) == 2: # "[]"
        return 0
    return len([int(x) for x in line[1:-1].split(',')])

def layer_size(line):
    if len(line) == 2:
        return 0
    return int(line[1:-1].split(',')[0])

t_start = time.time()
scores = pd.read_csv(sys.argv[2], error_bad_lines=False, sep=";")
params = pd.read_csv(sys.argv[3], error_bad_lines=False, sep=";")
print("Data loaded ({0}s)".format(time.time() - t_start))

if sys.argv[1] == "mlp":
    params['hidden_layer_count'] =\
        params['hidden_layer_sizes'].map(count_layers)
    params['hidden_layer_size'] =\
        params['hidden_layer_sizes'].map(layer_size)
    params['alpha'] = params['alpha'].map(lambda x: float(x))

    activation_color = {
        "identity": 'red',
        "logistic": 'green',
        "tanh": 'blue',
        "relu": 'black'
    }

    solver_color = {
        "lbfgs": 'red',
        "sgd": 'green',
        "adam": 'blue'
    }

    learning_rate_color = {
        "constant": 'red',
        "invscaling": 'green',
        "adaptive": 'blue'
    }

    fig = plt.figure(figsize=(12, 12))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0]\
                    for color in activation_color.values()]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(params['hidden_layer_size'],\
               params['hidden_layer_count'],\
               scores['f1'],\
               c=[activation_color[x] for x in params["activation"]])
    ax.set_xlabel("hidden layer size")
    ax.set_ylabel("hidden layer count")
    ax.set_zlabel("f1 score")
    plt.legend(handlelist, activation_color.keys(), loc='best')
    plt.title("MLP score based on activation function")
    plt.savefig('mlp_activation.png')

    fig = plt.figure(figsize=(12, 12))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0]\
                    for color in solver_color.values()]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(params['hidden_layer_size'],\
               params['hidden_layer_count'],\
               scores['f1'],\
               c=[solver_color[x] for x in params["solver"]])
    ax.set_xlabel("hidden layer size")
    ax.set_ylabel("hidden layer count")
    ax.set_zlabel("f1 score")
    plt.legend(handlelist, solver_color.keys(), loc='best')
    plt.title("MLP score based on solver function")
    plt.savefig('mlp_solver.png')

    fig = plt.figure(figsize=(12, 12))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0]\
                    for color in learning_rate_color.values()]
    ax = fig.add_subplot(111)
    ax.scatter(params['alpha'],\
               scores['f1'],\
               c=[learning_rate_color[x] for x in params["learning_rate"]])
    ax.set_xlabel("alpha")
    ax.set_ylabel("f1 score")
    plt.legend(handlelist, learning_rate_color.keys(), loc='best')
    plt.title("MLP score based on learning rate")
    plt.savefig('mlp_learning.png')

elif sys.argv[1] == "ada":
    params["base_estimator__max_depth"] =\
        params["base_estimator__max_depth"].map(lambda x: int(x))
    params["learning_rate"] =\
        params["learning_rate"].map(lambda x: float(x))
    params["n_estimators"] =\
        params["n_estimators"].map(lambda x: int(x))

    min_depth = min(params["base_estimator__max_depth"])
    max_depth = max(params["base_estimator__max_depth"])
    depth_colors =\
        { k:tuple(\
            [random.choice([x for x in range(256)]) / 255 for _ in range(3)]\
            ) for k in range(min_depth, max_depth + 1)}

    fig = plt.figure(figsize=(12, 12))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0]\
                    for color in depth_colors.values()]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(params['learning_rate'],\
               params['n_estimators'],\
               scores['f1'],\
               c=[depth_colors[x] for x in params["base_estimator__max_depth"]])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("n_estimators")
    ax.set_zlabel("f1 score")
    plt.legend(handlelist, depth_colors.keys(), loc='best')
    plt.title("AdaBoost f1 score and base estimator (DecisionTree) max depth")
    plt.savefig('adaboost_depth.png')
