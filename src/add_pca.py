import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import lightgbm as lgb
import gc
import datetime

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')


def get_now():
    now = datetime.datetime.now()
    return '{0:%Y-%m-%d %H:%M:%S}'.format(now)


def get_proba_lgbm(X, y, X_test, max_depth=3, num_leaves=7, is_sub=False):
    target = 'is_attributed'
    lgb_params = {
        'learning_rate': 0.10,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': num_leaves,  # 2^max_depth - 1
        'max_depth': max_depth,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }

    lgbtrain = lgb.Dataset(X, label=y,
                           categorical_feature=categorical
                           )
    if not is_sub:
        lgbvalid = lgb.Dataset(valid[features].values, label=valid[target].values,
                               categorical_feature=categorical
                               )
        valid_names = 'valid'
    else:
        lgbvalid = lgbtrain
        valid_names = 'train'

    evals_results = {}
    num_boost_round = 500
    early_stopping_rounds = 30

    booster = lgb.train(
        lgb_params,
        lgbtrain,
        valid_sets=[lgbvalid],
        valid_names=[valid_names],
        evals_result=evals_results,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=1
    )

    return booster.predict(X_test)


if __name__ == '__main__':
    # read files
    print('[{}]Start:Read train'.format(get_now()))
    train = pd.read_csv('../data/preprocesssed_train.csv.gz',
                        compression='gzip')
    print('[{}]Start:Read test'.format(get_now()))
    test = pd.read_csv('../data/preprocesssed_test.csv.gz',
                       compression='gzip')
    print('[{}]Start:Read valid'.format(get_now()))
    valid = pd.read_csv('../data/preprocesssed_val.csv.gz',
                        compression='gzip')
    print('[{}]Finished:Read All Data'.format(get_now()))
    print('Length test data:{}'.format(len(test)))

    # preprocess
    print('[{}]Start:Data Preprocessing'.format(get_now()))

    features = list(set(['app',
                         'app_by_channel_countuniq',
                         'app_by_channel_countuniq',
                         'channel',
                         'device',
                         'hour',
                         'ip_app_by_os_countuniq',
                         'ip_app_by_os_countuniq',
                         'ip_app_device_os_channel_nextClick',
                         'ip_app_device_os_channel_nextClick',
                         'ip_app_os_by_hour_var',
                         'ip_app_os_by_hour_var',
                         'ip_app_oscount',
                         'ip_app_oscount',
                         'ip_appcount',
                         'ip_appcount',
                         'ip_by_app_countuniq',
                         'ip_by_app_countuniq',
                         'ip_by_channel_countuniq',
                         'ip_by_channel_countuniq',
                         'ip_by_device_countuniq',
                         'ip_by_device_countuniq',
                         'ip_by_os_cumcount',
                         'ip_by_os_cumcount',
                         'ip_channel_prevClick',
                         'ip_channel_prevClick',
                         'ip_day_by_hour_countuniq',
                         'ip_day_by_hour_countuniq',
                         'ip_day_hourcount',
                         'ip_day_hourcount',
                         'ip_device_os_by_app_countuniq',
                         'ip_device_os_by_app_countuniq',
                         'ip_device_os_by_app_cumcount',
                         'ip_device_os_by_app_cumcount',
                         'ip_os_device_app_nextClick',
                         'ip_os_device_app_nextClick',
                         'ip_os_device_nextClick',
                         'ip_os_device_nextClick',
                         'ip_os_prevClick',
                         'ip_os_prevClick',
                         'os']))

    y_train = train['is_attributed']
    X_train = train[features]
    del train
    gc.collect()
    click_ids = test.click_id.values
    X_test = test[features]
    del test
    gc.collect()
    X_valid = valid[features]
    y_valid = valid['is_attributed']
    del valid
    gc.collect()

    len_X_train = len(X_train)
    len_X_test = len(X_test)
    len_X_valid = len(X_valid)

    y_train = y_train.astype(int)

    categorical = ['app', 'device', 'os', 'channel', 'hour']

    print('[{}]Start:merge train, test and valid'.format(get_now()))
    merge = pd.concat([X_train, X_test, X_valid], axis=0, ignore_index=True)
    del X_train, X_test, X_valid
    gc.collect()
    print('[{}]Finished:merge train, test and valid'.format(get_now()))

    print('[{}]Start:Type Changing int to str'.format(get_now()))
    categorical_df = merge[categorical].copy()
    categorical_df = categorical_df.astype(str)
    print('[{}]Finished:Type Changing int to str'.format(get_now()))

    print('[{}]Start:Selection dummies(using value_counts())'.format(get_now()))
    app_vc = categorical_df.app.value_counts().sort_values(ascending=False).index.tolist()[:1000]
    dev_vc = categorical_df.device.value_counts().sort_values(ascending=False).index.tolist()[:1000]
    os_vc = categorical_df.os.value_counts().sort_values(ascending=False).index.tolist()[:1000]
    cha_vc = categorical_df.channel.value_counts().sort_values(ascending=False).index.tolist()[:1000]
    print('[{}]Finished:Selection dummies(using value_counts())'.format(get_now()))
    print('[{}]Start:replaceTop1000'.format(get_now()))
    categorical_df.app = categorical_df.app.apply(lambda x: x if x in app_vc else '-1')
    categorical_df.device = categorical_df.device.apply(lambda x: x if x in dev_vc else '-1')
    categorical_df.os = categorical_df.os.apply(lambda x: x if x in os_vc else '-1')
    categorical_df.channel = categorical_df.channel.apply(lambda x: x if x in cha_vc else '-1')
    print('[{}]Finished:replaceTop1000'.format(get_now()))
    print('[{}]Start:get dummies'.format(get_now()))
    categorical_df = pd.get_dummies(categorical_df)
    merge.drop(categorical, axis=1, inplace=True)
    merge = pd.concat([merge, categorical_df], axis=1, ignore_index=True)
    print('[{}]Finished:get dummies'.format(get_now()))
    X_train = merge.iloc[:len_X_train, :]
    X_test = merge.iloc[len_X_train:len_X_train + len_X_test, :]
    X_valid = merge.iloc[len_X_train + len_X_test:, :]
    del merge, categorical_df
    gc.collect()

    print('[{}]Finished:Data Preprocessing'.format(get_now()))

    print('Length test data:{}'.format(len(X_test)))

    # final prediction
    print('[{}]Start:Final Prediction'.format(get_now()))
    sub = pd.DataFrame()
    sub['click_id'] = click_ids
    sub.click_id = sub.click_id.astype(int)
    sub['is_attributed'] = get_proba_lgbm(X_train.values, y_train.values, X_test.values)
    print('[{}]Finished:Final Prediction'.format(get_now()))

    # output sub
    print('[{}]Start:Output Submission Data'.format(get_now()))
    output = os.path.join(DATA_DIR, 'sub_lgb_semis_{0:%Y%m%d_%H%M%S}.csv.gz'.format(datetime.datetime.now()))
    sub.to_csv(output,
               index=False,
               compression='gzip',
               float_format='%.9f')
    print('[{}]Finished:Output Submission Data'.format(get_now()))
    print('[{}]Finished:All Process'.format(get_now()))
