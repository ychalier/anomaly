# anomaly
Classification anomaly detection in IOT with Machine Learning

## setup

use a virtual environment with *Python 3.6* and install the modules from [requirements.txt](requirements.txt).

    python -m virtualenv venv
	source venv/bin/activate
	pip install -r requirements.txt

## data prospection

 - [40 Brilliant Open Data Projects Preparing Smart Cities for 2018](https://carto.com/blog/forty-brilliant-open-data-projects-preparing-smart-cities-2018/)
 - [Sci-Hub](http://sci-hub.hk)
 - [Dweet](https://dweet.io/see)
 - [Engage](http://www.engagedata.eu/dataset-search/?q=)
 - [Intel Lab Data](http://db.csail.mit.edu/labdata/labdata.html)

## usage

 You only need two files, [detector.py](detector.py) contains the `Detector` that is used to store and compute datasets. A example of its usage is shown in [main.ipynb](main.ipynb).

  - [body.csv](http://devyss.byethost31.com/dl/body.csv) (source: [Grete Heinz, Louis J. Peterson, Roger W. Johnson, and Carter J. Kerk. Exploring relationships in body dimensions. Journal of Statistics Education, Volume 11, Number 2](http://ww2.amstat.org/publications/jse/v11n2/datasets.heinz.html))
  - [moto.csv](http://devyss.byethost31.com/dl/moto.csv) (sample from the [Motorcycle database](http://www.teoalida.com/cardatabase/motorcycles/) from [Teoalida](http://www.teoalida.com), which is a gathering of data scrapped of [bikez.com](www.bikez.com))
  - [moto2.csv](http://devyss.byethost31.com/dl/moto2.csv)

To perform tests that require higher computing performance, use [exec.py](exec.py) on a remote server. One may use `scp` to copy databases over.

    scp *.csv user@host:path/

## parameters tuning

In order to find the best parameters for each classifier, we use a grid search with cross validation to find the best set of parameters. The class `Detector` now has a method called `tune_parameters` which seeks the best set of selected parameters for a given classifier. This method is used in a small [Python script](parameters.py) that can be run on more powerful servers.

### search and results

For each classifier, we will report here the list of its parameters, the range of values we tried for each one of them and finally the value of the parameter which results in the *best global set of parameters* (i.e. it might not be the best value considering only this parameter but the value that is present in the best set of parameters).

#### AdaBoost

| parameter | description | range | *best* value |
| --- | --- | --- | --- |
| `base_estimator`   | The base estimator from which the boosted ensemble is built. Those often are *Decision Trees* with a different *maximum depth*. | ? | ?  |
| `n_estimators`  | The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. | 1-100  | ?  |
| `learning_rate`  | Learning rate shrinks the contribution of each classifier by `learning_rate`. | ?  | ?  |

#### Multi-Layered Perceptron

| parameter | description | range | *best* value |
| --- | --- | --- | --- |
| `hidden_layer_sizes`   |  The ith element represents the number of neurons in the ith hidden layer. |   |   |
| `activation`   | Activation function for the hidden layer.  | `['identity', 'logistic', 'tanh', 'relu']` |   |
| `solver`   | The solver for weight optimization. | `['lbfgs', 'sgd', 'adam']` |   |
| `alpha`   | L2 penalty (regularization term) parameter.  |   |   |
| `learning_rate`   |  Learning rate schedule for weight updates.  |  `['constant', 'invscaling', 'adaptive']`  |   |
