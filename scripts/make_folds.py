import json

import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':

    meta = pd.read_csv('data/kaggle/metadata_train.csv')
    folds = []

    ids = meta['id_measurement'].unique()
    targets = meta.groupby('id_measurement')['target'].apply(lambda x: ''.join(map(str, x)))
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fit_idx, val_idx in cv.split(ids, targets.loc[ids]):
        fit_ids, val_ids = ids[fit_idx], ids[val_idx]
        folds.append({
            'fit': {
                'measurement_ids': fit_ids.tolist(),
                'signal_ids': meta[meta['id_measurement'].isin(fit_ids)]['signal_id'].values.tolist()
            },
            'val': {
                'measurement_ids': val_ids.tolist(),
                'signal_ids': meta[meta['id_measurement'].isin(val_ids)]['signal_id'].values.tolist()
            }
        })

    with open('oof/folds.json', 'w') as file:
        json.dump(folds, file)
