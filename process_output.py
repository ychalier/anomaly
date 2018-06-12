def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
scores = pd.read_csv(sys.argv[1], error_bad_lines=False, sep=";")
params = pd.read_csv(sys.argv[2], error_bad_lines=False, sep=";")
params['hidden_layer_count'] = params['hidden_layer_sizes'].map(count_layers)
params['hidden_layer_size'] = params['hidden_layer_sizes'].map(layer_size)
print("Data loaded ({0}s)".format(time.time() - t_start))

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
