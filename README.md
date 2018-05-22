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

  - [body.csv](http://devyss.byethost31.com/dl/body.csv) ([source](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiRhObqy4fbAhXKWhQKHYACC48QFggrMAA&url=http%3A%2F%2Feric.univ-lyon2.fr%2F~ricco%2Ftanagra%2Ffichiers%2Fbody.xls&usg=AOvVaw1j0Zq5sAnaMPNaAXCcDjws))
  - [moto.csv](http://devyss.byethost31.com/dl/moto.csv) (sample from [Teoalida](http://www.teoalida.com/cardatabase/motorcycles/))
  - [moto2.csv](http://devyss.byethost31.com/dl/moto2.csv)
