import glob
import json
import os

import numpy as np
from scipy import cluster
import tqdm


def is_neg_to_pos_crossing(sig, candidate):
    i = candidate
    before = sig.take(range(i - 10000, i), mode='wrap')
    after = sig.take(range(i, i + 10000), mode='wrap')
    return before.mean() < 0 and after.mean() > 0


def find_0_crossings(sig):
    """Returns all the indexes where the signal crosses over or under 0."""
    return np.where(np.diff(np.sign(sig)))[0]


def find_origin(sig):
    crossings = find_0_crossings(sig)

    # Because of the nature of the problem we know there are 2 crossings over 0. But because there
    # is a lot of noise there seem to be many crossings over 0. We use k-means clustering with k=2
    # to approximately determine where these 2 crossings are.
    a, b = cluster.vq.kmeans(crossings.astype(float), 2, iter=2, thresh=1e-05)[0]
    a, b = int(a), int(b)

    # One of the two crossings is the one where the signal goes from the negative area to
    # the positive area
    if is_neg_to_pos_crossing(sig, a):
        return a
    return b


if __name__ == '__main__':

    origins = {}

    for path in tqdm.tqdm(glob.glob('data/signals/*/*.npy'), unit='signal'):

        # Extract the signal ID from the path
        signal_id = int(os.path.basename(path).split('.')[0])

        # Load the signal
        signal = np.load(path)

        # Find the signal's origin
        origins[signal_id] = find_origin(signal)

    # Save the origins
    with open('data/signal_origins.json', 'w') as file:
        json.dump(origins, file)

    print('\a')
