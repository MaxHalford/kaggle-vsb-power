import pandas as pd


files = [
    'binary_lgb_0.842_0.035_0.632_0.034.csv',
    'multi_lgb_0.969_0.015_0.650_0.039.csv',
    'submission.csv'
]

predictions = {}

for file in files:
    name = file.split('.')[0].split('_')[0]
    sub = pd.read_csv(file)
    predictions[name] = sub['target']

predictions = pd.DataFrame.from_dict(predictions)
predictions.index = sub['signal_id']

print(predictions.corr())

predictions.mode(axis=1)[0].to_frame('target').to_csv('blend.csv')
