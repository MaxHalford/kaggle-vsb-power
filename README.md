# Kaggle VSB power line fault detection

This is my solution to the [VSB Power Line Fault Detection](https://www.kaggle.com/competitions/vsb-power-line-fault-detection/overview) competition. My solution is very standard and consists in manually extracting features before feeding them to LightGBM. This worked quite well at first and I managed to reach the top 10 of the competition. However RNNs seem to be the right way to go, but I'm not very interested in deep learning. Usually I don't upload Kaggle solutions that didn't do well, but I'm making an exception for this one as I'm quite satisfied with the feature extraction pipeline I put in place. If you want to run the code make sure you are using Python 3 and have installed the dependencies listed in the `requirements.txt` file.

## Splitting the signals

```sh
>>> python scripts/split_signals.py
```

The data provided by the competition is stored in an [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file. Reading from the HDF5 file is anything but fast. My idea was to first split the signals into separate numpy files using the `numpy.load` method. Loading the signals using `numpy.save` then takes something in the range of microseconds. This is extremely important because throughout the competition the data will be loaded in memory many times.


## Aligning the signals

```sh
>>> python scripts/find_signal_origins.py
```

Although each signal represents one period of an electrical sine wave, they don't all start at the same time. I decided to align them so that they all started from 0 and started by going upwards. This could be useful as some features could be based on a particular region of the signal. I didn't really exploit this as I gave up the competition when RNNs arrived. To align the signals I used a simple method which starts by searching for the two points where the signal crosses 0. Because there is a lot of noise I used a k-means clustering scheme with `k = 2` to approximate the two positions. I then decided which of the two crossings was the one that I wanted by looking left and right from both crossings.


## Extracting features

```sh
>>> python scripts/extract_solo_features.py
```

I won't go into detail about which features I extracted as I'm sure some people did better and will talk about it when the competition is over. The only thing I want to mention is *how* I extracted the features. As mentioned above the signals were split into separate `.npy` file using `numpy` and were stored in the `data` directory. I then simply looped over the files and extracted the features in parallel using a `ThreadPoolExecutor` from the `concurrent.futures` module from Python's standard library. The trick is that before computing the features I first loaded the ones that had already been computed so that I had didn't recompute them unnecessarily. This is definitely not rocket science but I thought the code to be quite concise and rather readable so I deemed it worthy of being shared online.


## Cross-validation folds

```sh
>>> python scripts/make_folds.py
```

I like generating CV folds before doing the machine learning. I save these as a JSON file called `folds.json` in the `oof` directory and load them during the machine learning phase. This is practical because you can share the folds with others and use them with multiple models. I made sure that the folds didn't "leak" by putting signals from the same measurement in the training folds as well as the validation folds.


## Machine learning

I started by trying to learn the labels of each signals individually. I then converted the problem to a multi-class classification problem by joining the signals of each measurement together. Because each signal has a binary label and there were 3 signals per label this resulted in an `2^3 = 8` class problem. What's more by permuting the 3 signals I was able to augment the multi-label dataset by a factor of `3! = 6`. The code is available in the `Solution.ipynb` notebook. I didn't comment it but it should be readable. In the end this will produce a submission in the `submissions` directory and out-of-fold predictions in the `oof` directory.
