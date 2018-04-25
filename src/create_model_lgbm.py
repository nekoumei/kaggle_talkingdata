import os
import pandas as pd
import gc
import lightgbm as lgb
import datetime

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')
MODEL_PATH = os.path.join(PROJECT_ROOT,'model')


def create_model(merge,
                 lgb_params,
                 categorical_features=['ip', 'app', 'os', 'channel', 'device', 'day'],
                 predictors=[]
                 ):
    X = merge[predictors].values
    y = merge['is_attributed'].values
    #predictors=list(set(merge.columns) - set(['attributed_time', 'click_time', 'is_attributed']))

    lgbtrain = lgb.Dataset(X, label=y,
                           feature_name=predictors,
                           categorical_feature=categorical_features
                           )

    evals_results = {}
    num_boost_round = 200
    early_stopping_rounds = 30

    booster = lgb.train(
        lgb_params,
        lgbtrain,
        valid_sets=[lgbtrain],
        valid_names=['train'],
        evals_result=evals_results,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=1
    )

    return booster, booster.best_iteration


def preprocess(df):
    df['day'] = df.click_time.dt.day
    df['hour'] = df.click_time.dt.hour
    df['minute'] = df.click_time.dt.minute
    df['second'] = df.click_time.dt.second

    return df

def get_now():
    now = datetime.datetime.now()
    return '{0:%Y-%m-%d %H:%M:%S}'.format(now)


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


if __name__ == '__main__':
    for bagging_i in range(10):
        print('[{}]Start:BaggingNo_{}'.format(get_now(), bagging_i))
        print('[{}]Start:All data preparing'.format(get_now()))
        print('[{}]Start:read positive'.format(get_now()))
        positive = pd.read_csv(os.path.join(DATA_DIR, 'train_positive.csv'), parse_dates=['click_time'])
        print('[{}]Start:read negative'.format(get_now()))
        negative = pd.read_csv(os.path.join(DATA_DIR, 'train_negative.csv'), parse_dates=['click_time'])
        negative_sampled = negative.sample(10000000)
        del negative
        gc.collect()

        print('[{}]Finished:All data preparing'.format(get_now()))

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
            'metric':'auc',

            'learning_rate': 0.15,
            'num_leaves': 63,  # 2^max_depth - 1
            'max_depth': 6,  # -1 means no limit
            'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 100,  # Number of bucketed bin for feature values
            'subsample': 0.7,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'scale_pos_weight': 99
        }
        merge = pd.concat([positive, negative_sampled])
        merge = preprocess(merge)
        print('[{}]Start:training'.format(get_now()))
        booster = create_model(merge, lgb_params)
        print('[{}]Finished:training'.format(get_now()))

        predict(booster)

        print('[{}]Finished:BaggingNo_{}'.format(get_now(), bagging_i))

    print('[{}]Finished:All Process'.format(get_now()))