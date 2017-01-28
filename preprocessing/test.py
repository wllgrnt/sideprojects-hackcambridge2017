import pandas as pd
import numpy as np
from data_cleaning import *
from feature_scaling import *

df = pd.DataFrame.from_csv('../DataFiles/train.csv')

X = df['Attr1'].values
nans = np.isnan(X)
print(np.sum(nans))

df = clean_data(df)
X = df['Attr1'].values
nans = np.isnan(X)
print(np.sum(nans))

df = scale_features(df)
