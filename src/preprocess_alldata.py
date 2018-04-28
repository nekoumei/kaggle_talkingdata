import os
import numpy as np
import pandas as pd
import gc
import datetime
import create_model_lgbm as cml

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')
MODEL_PATH = os.path.join(PROJECT_ROOT,'model')

def get_now():
    now = datetime.datetime.now()
    return '{0:%Y-%m-%d %H:%M:%S}'.format(now)


def preprocess_baris(df, bagging_mode):
    df['day'] = df.click_time.dt.day
    df['hour'] = df.click_time.dt.hour
    df['minute'] = df.click_time.dt.minute
    df['second'] = df.click_time.dt.second

    naddfeat = 9
    for i in range(0, naddfeat):
        if i == 0:
            selcols = ['ip', 'channel']
            QQ = 4
        if i == 1:
            selcols = ['ip', 'device', 'os', 'app']
            QQ = 5
        if i == 2:
            selcols = ['ip', 'day', 'hour']
            QQ = 4
        if i == 3:
            selcols = ['ip', 'app']
            QQ = 4
        if i == 4:
            selcols = ['ip', 'app', 'os']
            QQ = 4
        if i == 5:
            selcols = ['ip', 'device']
            QQ = 4
        if i == 6:
            selcols = ['app', 'channel']
            QQ = 4
        if i == 7:
            selcols = ['ip', 'os']
            QQ = 5
        if i == 8:
            selcols = ['ip', 'device', 'os', 'app']
            QQ = 4

        print('selcols', selcols, 'QQ', QQ)

        filename = 'X%d.csv' % (i)

        if (os.path.exists(filename)) and not bagging_mode:
            if QQ == 5:
                gp = pd.read_csv(filename, header=None)
                df['X' + str(i)] = gp
            else:
                gp = pd.read_csv(filename)
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
        else:
            if QQ == 0:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].count().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 1:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].mean().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 2:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].var().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 3:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].skew().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 4:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].nunique().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                df = df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 5:
                gp = df[selcols].groupby(by=selcols[0:len(selcols) - 1])[selcols[len(selcols) - 1]].cumcount()
                df['X' + str(i)] = gp.values

            gp.to_csv(filename, index=False)

        del gp
        gc.collect()

    print('doing nextClick')
    predictors = []

    new_feature = 'nextClick'
    filename = 'nextClick.csv'

    if os.path.exists(filename) and not bagging_mode:
        print('loading from save file')
        QQ = pd.read_csv(filename).values
    else:
        D = 2 ** 26
        df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df[
            'device'].astype(str) \
                                + "_" + df['os'].astype(str)).apply(hash) % D
        click_buffer = np.full(D, 3000000000, dtype=np.uint32)

        df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, t in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
            next_clicks.append(click_buffer[category] - t)
            click_buffer[category] = t
        del click_buffer
        QQ = list(reversed(next_clicks))

        print('saving')
        pd.DataFrame(QQ).to_csv(filename, index=False)

    df[new_feature] = QQ
    predictors.append(new_feature)

    df[new_feature + '_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature + '_shift')

    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    df = df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    df = df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])[
        ['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    df = df.merge(gp, on=['ip', 'day', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(
        index=str, columns={'hour': 'ip_app_os_var'})
    df = df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[
        ['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    df = df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[
        ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    df = df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    df.info()
    df['ip_tcount'] = df['ip_tcount'].astype('uint16')
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')

    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'minute', 'second', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour'])
    for i in range(0, naddfeat):
        predictors.append('X' + str(i))

    print('predictors', predictors)

    return df, predictors

def predict(booster):
    reader = pd.read_csv(os.path.join(DATA_DIR, 'test.csv.zip'),
                         parse_dates=['click_time'],
                         chunksize=1000000,
                         compression='zip')
    output = os.path.join(DATA_DIR, 'submission_lgb_{0:%Y%m%d_%H%M%S}.csv'.format(datetime.datetime.now()))
    for i, test in enumerate(reader):
        print('[{}]Start:Preprocessing Data:Size:{}'.format(get_now(), len(test)))
        test = preprocess(test)
        print('[{}]Finish:Preprocessing Data:Size:{}'.format(get_now(), len(test)))

        print('[{}]Start:Predicting Data'.format(get_now()))
        X = test.drop(['click_id', 'click_time'], axis=1)
        y_prob = booster.predict(X.values)
        print('[{}]Finish:Predicting Data'.format(get_now()))

        print('[{}]Start:output Data'.format(get_now()))
        y = pd.DataFrame({
            'click_id': test['click_id'],
            'is_attributed': y_prob
        })

        if i == 0:
            if os.path.isfile(output):
                os.remove(output)
            header = True
        else:
            header = False
        y.to_csv(output, index=False, header=header, mode='a')
        print('[{}]Finish:output Data'.format(get_now()))

    print('[{}]Finish:Output submission data'.format(get_now()))


def execute(bagging_mode, bagging_dir):
    debag_mode = False
    if debag_mode:
        print('[{}]Start:Small data preparing(DebagMode)'.format(get_now()))
        print('[{}]Start:read train'.format(get_now()))
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv.zip'), parse_dates=['click_time'], nrows=1000, compression='zip')
        print('[{}]Start:read test'.format(get_now()))

        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv.zip'),
                              parse_dates=['click_time'],
                              compression='zip',
                              nrows=1000)
    else:
        print('[{}]Start:All data preparing'.format(get_now()))
        print('[{}]Start:read train'.format(get_now()))
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv.zip'), parse_dates=['click_time'], compression='zip')
        print('[{}]Start:read test'.format(get_now()))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv.zip'),
                              parse_dates=['click_time'],
                              compression='zip'
                               )

    len_train = len(train_df)

    print('[{}]Finished:All data preparing'.format(get_now()))

    merge = pd.concat([train_df, test_df])
    del test_df
    gc.collect()
    merge, predictors = preprocess_baris(merge, bagging_mode)

    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    test_df = merge[len_train:]
    merge = merge[:len_train]

    print("train size: ", len(merge))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    # Setting Parameters
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric': 'auc',

        'learning_rate': 0.15,
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 99
    }
    print('[{}]Start:training'.format(get_now()))
    booster, best_iteration = cml.create_model(merge,
                               lgb_params,
                               categorical_features=categorical,
                               predictors=predictors)
    print('[{}]Finished:training'.format(get_now()))

    print('[{}]Start:predicting'.format(get_now()))
    sub['is_attributed'] = booster.predict(test_df[predictors].values, num_iteration=best_iteration)
    print('[{}]Finished:predicting'.format(get_now()))
    print('[{}]Start:output submission'.format(get_now()))
    if bagging_mode:
        output = os.path.join(DATA_DIR,
                              bagging_dir,
                              'submission_lgb_{0:%Y%m%d_%H%M%S}.csv.gz'.format(datetime.datetime.now()))
    else:
        output = os.path.join(DATA_DIR, 'submission_lgb_{0:%Y%m%d_%H%M%S}.csv.gz'.format(datetime.datetime.now()))
    sub.to_csv(output, index=False, compression='gzip')
    print('[{}]Finished:output submission'.format(get_now()))
    print('[{}]Finished:All Process'.format(get_now()))


if __name__ == '__main__':
    bagging_mode = True

    if bagging_mode:
        bagging_dir = 'bagging_{0:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
        os.mkdir(os.path.join(DATA_DIR, bagging_dir))
        for i in range(10):
            print('[{}]Start:Bagging Process:{}'.format(get_now(), i))
            execute(bagging_mode, bagging_dir)
            print('[{}]Finished:Bagging Process:{}'.format(get_now(), i))
    else:
        execute(bagging_mode)