import os
import pandas as pd
import datetime

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')


def get_now():
    now = datetime.datetime.now()
    return '{0:%Y-%m-%d %H:%M:%S}'.format(now)


print('[{}]Start:Read sub'.format(get_now()))
test = pd.read_csv('../data/sub_lgb_semis_20180430_225353.csv.gz',
                   compression='gzip')
print('[{}]Finished:Read test'.format(get_now()))
test.click_id = test.click_id.astype(int)

print('[{}]Start:output sub'.format(get_now()))
test.to_csv('../data/sub_lgb_semis_20180430_225353_rev.csv.gz',
            index=False,
            compression='gzip')
print('[{}]Finished:Output sub'.format(get_now()))
