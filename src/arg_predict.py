import os
import pandas as pd
import gc
import datetime
from pathlib import Path

PROJECT_ROOT = Path('.') / '..'
DATA_DIR = PROJECT_ROOT / 'data'
BAGGING_DIR = DATA_DIR / 'bagging_20180426_002417'

# csvs = [predicted for predicted in BAGGING_DIR.glob('*.csv.gz')]
csvs = [
    DATA_DIR / 'sub_bai_6.csv.gz',
    DATA_DIR / 'sub_lgb_semis_20180501_124039.csv.gz'
    DATA_DIR / 'submission_lgb_20180428_160430.csv.gz'
]
print(BAGGING_DIR.resolve())
merge = pd.DataFrame()
for i, predicted_csv in enumerate(csvs):
    pred_df = pd.read_csv(predicted_csv, compression='gzip')
    if i == 0:
        merge['click_id'] = pred_df['click_id']
    merge = pd.concat([merge, pred_df['is_attributed']], axis=1)
    print(merge.head())

calc_df = merge.drop('click_id', axis=1)
calc_df = calc_df.mean(axis=1)
merge = pd.concat([merge['click_id'], calc_df], axis=1)
merge.columns = ['click_id', 'is_attributed']
now = datetime.datetime.now()
merge.to_csv('../data/submission_lgb_bugged_{0:%Y%m%d_%H%M%S}.csv.gz'.format(now),
             compression='gzip',
             index=False
             )