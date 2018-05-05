import os
import pandas as pd
import numpy as np
import gc
import datetime

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')


def get_now():
    now = datetime.datetime.now()
    return '{0:%Y-%m-%d %H:%M:%S}'.format(now)


def read_preprocessed_csvs(debug=True):
    # read files
    if not debug:
        print('[{}]Start:Read train'.format(get_now()))
        train = pd.read_csv('../data/preprocesssed_train.csv.gz',
                            compression='gzip')
        print('[{}]Start:Read test'.format(get_now()))
        test = pd.read_csv('../data/preprocesssed_test.csv.gz',
                           compression='gzip')
        print('[{}]Start:Read valid'.format(get_now()))
        valid = pd.read_csv('../data/preprocesssed_val.csv.gz',
                            compression='gzip')
    else:
        print('[{}]Start:Debug:Read train'.format(get_now()))
        train = pd.read_csv('../data/preprocesssed_train.csv.gz',
                            nrows=10000,
                            compression='gzip')
        print('[{}]Start:Debug:Read test'.format(get_now()))
        test = pd.read_csv('../data/preprocesssed_test.csv.gz',
                           nrows=10000,
                           compression='gzip')
        print('[{}]Start:Read valid'.format(get_now()))
        valid = pd.read_csv('../data/preprocesssed_val.csv.gz',
                            compression='gzip')

    print('[{}]Finished:Read All Data'.format(get_now()))
    print(f'train length: {len(train)}')
    print(f'test length: {len(test)}')
    print(f'valid length: {len(valid)}')

    return train, test, valid