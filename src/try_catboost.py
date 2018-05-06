import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from catboost import CatBoostClassifier
import gc
import datetime
from read_csv import get_now, read_preprocessed_csvs

PROJECT_ROOT = os.path.join(os.getcwd(), '..')
DATA_DIR = os.path.join(PROJECT_ROOT,'data')

train, test, valid = read_preprocessed_csvs(debug=False)

# separate X y
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
X_train = train[features]
y_train = train['is_attributed']
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
columns_list = X_train.columns.tolist()
categorical = ['app', 'device', 'os', 'channel', 'hour']
categorical_idx = [columns_list.index(cat) for cat in categorical]
print(categorical_idx)
print(X_train.columns.tolist())
cbc = CatBoostClassifier(
    iterations=300,
    learning_rate=0.3,
    eval_metric='AUC',
    scale_pos_weight=200,
    use_best_model=True,
    max_depth=3,
    bootstrap_type='Bernoulli',
    subsample=0.7
)
cbc.fit(
    X_train,
    y_train,
    cat_features=categorical_idx,
    eval_set=(X_valid, y_valid),
    verbose_eval=True
)

# final prediction
print('[{}]Start:Final Prediction'.format(get_now()))
sub = pd.DataFrame()
sub['click_id'] = click_ids
sub.click_id = sub.click_id.astype(int)
sub['is_attributed'] = cbc.predict_proba(X_test).T[1]
print('[{}]Finished:Final Prediction'.format(get_now()))
print(sub.head())

# output sub
print('[{}]Start:Output Submission Data'.format(get_now()))
output = os.path.join(DATA_DIR, 'sub_cat_{0:%Y%m%d_%H%M%S}.csv.gz'.format(datetime.datetime.now()))
sub.to_csv(output,
           index=False,
           compression='gzip',
           float_format='%.9f')
print('[{}]Finished:Output Submission Data'.format(get_now()))
print('[{}]Finished:All Process'.format(get_now()))
