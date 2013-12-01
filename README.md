##[Bird Species Classification](http://www.kaggle.com/c/multilabel-bird-species-classification-nips2013)
- Second place in [Kaggle contest](http://www.kaggle.com/c/multilabel-bird-species-classification-nips2013/leaderboard)
- Les Bricoleurs: [Dima Kamalov](http://www.kaggle.com/users/135630/), [Matt Wescott](http://www.kaggle.com/users/16360)

####Hardware / OS

- M3.2XLarge instance on Amazon EC2
    - 8 virtual cpu
    - 26 gb ram
    - Intel Xeon E5-2670
- Ubuntu 12.04, 64 bit, Server Edition

####Dependencies
- numpy/scipy
- pandas
- scikit-learn

####Installation Instructions (from fresh EC2 instance)
1. `sudo apt-get update`
2. `sudo apt-get install git`
3.  `sudo apt-get install python-pip python-dev build-essential libblas-dev liblapack-dev gfortran`
4. `sudo pip install virtualenv`
5.  `git clone git@github.com:mattwescott/bird-recognition.git`
6. `cd bird-recognition`
7. Get the [data](http://www.kaggle.com/c/multilabel-bird-species-classification-nips2013/data)
8. Edit paths in SETTINGS.json to point to the data
9. `virtualenv venv`
10. `source venv/bin/activate`
11. `pip install -r requirements.txt`

####Generate Kaggle submission
- `python create_submission.py` (~4 hours runtime)

####Predict on new test set
- Not implemented
