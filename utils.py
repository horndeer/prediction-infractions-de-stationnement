import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import spearmanr
import numpy as np

def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def load_train_data():
    x_train = pd.read_csv("data/x_train_final_asAbTs5.csv", index_col=0)
    y_train = pd.read_csv("data/y_train_final_YYyFil7.csv", index_col=0)
    return x_train, y_train

def load_test_data():
    x_test = pd.read_csv("data/x_test_final_fIrnA7Q.csv", index_col=0)
    return x_test

def geo_filter(data, latitude_bounds=(0.995, 0.998), longitude_bounds=(0.998, 1.000)):
    return data[(data["latitude_scaled"] >= latitude_bounds[0]) & (data["latitude_scaled"] <= latitude_bounds[1]) & (data["longitude_scaled"] >= longitude_bounds[0]) & (data["longitude_scaled"] <= longitude_bounds[1])]

def total_count_filter(data, n_min=10):
    return data[data["total_count"] >= n_min]