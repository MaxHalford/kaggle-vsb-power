import collections
import concurrent.futures
import glob
import json
import math
import multiprocessing as mp
import os

import numpy as np
import pywt
from scipy import signal
from scipy import stats
import tqdm


def rolling_window(arr, window):
    """Returns an array view that can be used to calculate rolling statistics.

    From http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html.
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def mad(x, axis=None):
    return np.mean(np.abs(x - np.mean(x, axis)), axis)


def wavelet_denoise(x, wavelet='db1', mode='hard'):

    # Extract approximate and detailed coefficients
    c_a, c_d = pywt.dwt(x, wavelet)

    # Determine the threshold
    sigma = 1 / 0.6745 * mad(np.abs(c_d))
    threshold = sigma * math.sqrt(2 * math.log(len(x)))

    # Filter the detail coefficients
    c_d_t = pywt.threshold(c_d, threshold, mode=mode)

    # Reconstruct the signal
    y = pywt.idwt(np.zeros_like(c_a), c_d_t, wavelet)

    return y


def peaks(x):
    y = wavelet_denoise(x)
    peaks, properties = signal.find_peaks(y)
    widths = signal.peak_widths(y, peaks)[0]
    prominences = signal.peak_prominences(y, peaks)[0]
    return {
        'count': peaks.size,
        'width_mean': widths.mean() if widths.size else -1.,
        'width_max': widths.max() if widths.size else -1.,
        'width_min': widths.min() if widths.size else -1.,
        'prominence_mean': prominences.mean() if prominences.size else -1.,
        'prominence_max': prominences.max() if prominences.size else -1.,
        'prominence_min': prominences.min() if prominences.size else -1.,
    }


def denoised_std(x):
    return np.std(wavelet_denoise(x))


def signal_entropy(x):

    y = wavelet_denoise(x)

    for i in range(3):
        max_pos = y.argmax()
        y[max_pos - 1000:max_pos + 1000] = 0.

    return stats.entropy(np.histogram(y, 15)[0])


def detail_coeffs_entropy(x, wavelet='db1'):

    c_a, c_d = pywt.dwt(x, wavelet)

    return stats.entropy(np.histogram(c_d, 15)[0])


def bucketed_entropy(x):

    y = wavelet_denoise(x)

    return {
        f'bucket_{i}': stats.entropy(np.histogram(bucket, 10)[0])
        for i, bucket in enumerate(np.split(y, 10))
    }


FUNCS = [
    np.mean,
    np.std,
    stats.kurtosis,
    peaks,
    denoised_std,
    signal_entropy,
    detail_coeffs_entropy,
    bucketed_entropy
]


def compute_features(signal_path, origin, funcs):
    """Loads a signal, computes features, and adds them to a result queue."""
    signal = np.load(signal_path)
    signal = np.roll(signal, 800000 - origin)
    features = {f.__name__: f(signal) for f in funcs}
    return features


if __name__ == '__main__':

    # Load the existing features if there are any
    features = collections.defaultdict(dict)
    if os.path.exists('features/solo.json'):
        with open('features/solo.json', 'r') as file:
            for k, v in json.load(file).items():
                features[k] = v

    # Load the signal origins
    signal_origins = json.load(open('data/signal_origins.json'))

    jobs = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() // 2) as executor:

        try:

            for path in glob.glob('data/signals/*/*.npy'):

                # Extract the signal ID from the path
                signal_id = os.path.basename(path).split('.')[0]

                # Determine which features are missing
                funcs = [func for func in FUNCS if func.__name__ not in features.get(signal_id, {}).keys()]

                # No need to do anything whatsoever if all the features have been computed
                if funcs:
                    origin = signal_origins[signal_id]
                    jobs[executor.submit(compute_features, path, origin, funcs)] = signal_id

            # Now we simply go through the queue
            for job in tqdm.tqdm(concurrent.futures.as_completed(jobs), total=len(jobs), unit='signal'):
                signal_id = jobs[job]
                signal_features = job.result()
                for name, feature in signal_features.items():
                    features[signal_id][name] = feature

        except KeyboardInterrupt:
            executor.shutdown(wait=False)

    # Remove the deprecated features
    to_keep = set([func.__name__ for func in FUNCS])
    for signal_id, signal_features in features.items():
        for name in set(signal_features.keys()).difference(to_keep):
            features[signal_id].pop(name)

    # Save the features
    try:
        json.dumps(features)
    except TypeError as e:
        raise e
    with open('features/solo.json', 'w') as file:
        json.dump(features, file)

    print('\a')
