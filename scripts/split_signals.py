import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm


if __name__ == '__main__':

    meta_paths = ['data/kaggle/metadata_train.csv', 'data/kaggle/metadata_test.csv']
    signals_paths = ['data/kaggle/train.parquet', 'data/kaggle/test.parquet']

    for meta_path, signals_path in zip(meta_paths, signals_paths):

        meta = pd.read_csv(meta_path, dtype={'signal_id': str})

        for measurement_id, signal_ids in tqdm.tqdm(meta.groupby('id_measurement')['signal_id']):

            if os.path.exists(f'data/signals/{measurement_id}'):
                continue
            os.mkdir(f'data/signals/{measurement_id}')

            # Load the signals from the Parquet file
            signals = pq.read_pandas(signals_path, columns=signal_ids.values.tolist()).to_pandas()

            # Save each signal to a numpy file
            for signal_id in signal_ids:
                np.save(f'data/signals/{measurement_id}/{signal_id}', signals[signal_id].values)
