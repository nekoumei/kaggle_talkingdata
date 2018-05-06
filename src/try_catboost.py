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

train, test, valid = read_preprocessed_csvs(debug=True)

cbc = CatBoostClassifier()