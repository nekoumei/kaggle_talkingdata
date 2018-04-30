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


def pseudo_labeling(X_train, y_train, X_test, max_iter=1, th_confidence=0.95):
    """
    Extract test data with enough confidence and conduct pseudo-labeling.

    -----Parameters-----
    X_train:       Features of training data. Pandas DataFrame is expected
    y_train:       Target of training data. Pandas DataFrame is expected.
    X_test:        Features of test data. Pandas DataFrame is expected.
    max_iter:      Maximun number for iteration of pseudo-labeling. int is expected.
    th_confidence: Threshold of the confidence. float is expected.
    fit_params:    Parameters for "fit" method. dict is expected.

    -----Returns-----
    X_conf:        Features of test data with enough confidence.
    y_conf:        Target of test data. Because they have enough confidence, you can treat them as data with response==1.
    """

    continu = 1
    X_conf = DataFrame()
    y_conf = DataFrame()

    y_train = DataFrame(
        y_train)  # Although y_train is expected to be a DataFrame, we tends to input a Series for it.

    for iter_ in range(max_iter):

        if continu > 0:
            if iter_ > 0:
                X = pd.concat([X_train, X_conf], axis=0)
                y = pd.concat([y_train, y_conf], axis=0)

            if iter_ == 0:
                X = X_train
                y = y_train

            if X_test.shape[0] == 0:
                break
            print("Processing " + str(iter_ + 1) + " iteration")

            # fit_params = fit_params if fit_params is not None else {}
            # self.estimater.fit(X.as_matrix(), y.iloc[:,0].as_matrix(), **fit_params)

            # df_prob = DataFrame(self.estimater.predict_proba(X_test.as_matrix())[:,1],
            #                   index=X_test.index,
            #                   columns=["probability"])
            prob_arr = get_proba_lgbm(X.as_matrix(), y.iloc[:, 0].as_matrix(), X_test.as_matrix())
            df_prob = DataFrame(prob_arr,
                                index=X_test.index,
                                columns=["probability"])

            conf_index = df_prob[df_prob.probability > th_confidence].index
            conf2_index = df_prob[df_prob.probability < 1 - th_confidence].index
            print(conf_index)
            print(conf2_index)

            X_conf_ = X_test[X_test.index.isin(conf_index)]
            X_conf2_ = X_test[X_test.index.isin(conf2_index)]

            y_conf_ = DataFrame(index=X_conf_.index)
            y_conf_.index.names = y_train.index.names
            y_conf_[y_train.columns[0]] = 1

            y_conf2_ = DataFrame(index=X_conf2_.index)
            y_conf2_.index.names = y_train.index.names
            y_conf2_[y_train.columns[0]] = 0

            X_conf = pd.concat([X_conf, X_conf_, X_conf2_], axis=0)
            y_conf = pd.concat([y_conf, y_conf_, y_conf2_], axis=0)

            X_test.drop(conf_index, axis=0, inplace=True)
            X_test.drop(conf2_index, axis=0, inplace=True)

            continu = X_conf_.shape[0] + X_conf2_.shape[0]

            print(str(continu) + " samples with enough confidence were found at this iteration.")

    del X, y, X_train, y_train, X_test, X_conf_, y_conf_, X_conf2_, y_conf2_
    print("Finished!")

    return X_conf, y_conf


def get_proba_lgbm(X, y, X_test):
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    lgb_params = {
        'learning_rate': 0.10,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }

    lgbtrain = lgb.Dataset(X, label=y,
                           feature_name=features,
                           categorical_feature=categorical
                           )
    lgbvalid = lgb.Dataset(valid[features].values, label=valid[target].values,
                           feature_name=features,
                           categorical_feature=categorical
                           )

    evals_results = {}
    num_boost_round = 200
    early_stopping_rounds = 30

    booster = lgb.train(
        lgb_params,
        lgbtrain,
        valid_sets=[lgbvalid],
        valid_names=['valid'],
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

    y_train_gen = train['is_attributed']
    train = train[features]
    test = test[features]

    y_train_gen = y_train_gen.astype(int)

    print('[{}]Finished:Data Preprocessing'.format(get_now()))

    # execute semi-supervised learning
    print('[{}]Start:Semi-Supervised Learning'.format(get_now()))
    X_conf, y_conf = pseudo_labeling(train.copy(), y_train_gen.copy(), test.copy())
    print('[{}]Finished:Semi-Supervised Learning'.format(get_now()))
    print('[{}]Start:Prepare Data For Final Prediction'.format(get_now()))
    X_merged = pd.concat([train, X_conf], axis=0, ignore_index=True)
    y_merged = pd.concat([y_train_gen, y_conf.iloc[:, 0]], axis=0, ignore_index=True)
    print('[{}]Finished:Prepare Data For Final Prediction'.format(get_now()))

    # final prediction
    print('[{}]Start:Final Prediction'.format(get_now()))
    print('[{}]Start:Read test'.format(get_now()))
    test = pd.read_csv('../data/preprocesssed_test.csv.gz',
                       compression='gzip')
    test = test[features]
    print('[{}]Finished:Read test'.format(get_now()))
    click_ids = test.click_id.values

    sub = pd.DataFrame()
    sub['click_id'] = click_ids
    sub['is_attributed'] = get_proba_lgbm(X_merged.values, y_merged.values, test.values)
    print('[{}]Finished:Final Prediction'.format(get_now()))

    # output sub
    print('[{}]Start:Output Submission Data'.format(get_now()))
    output = os.path.join(DATA_DIR, 'sub_lgb_semis_{0:%Y%m%d_%H%M%S}.csv.gz'.format(datetime.datetime.now()))
    sub.to_csv(output,
               index=False,
               compression='gzip')
    print('[{}]Finished:Output Submission Data'.format(get_now()))
    print('[{}]Finished:All Process'.format(get_now()))
