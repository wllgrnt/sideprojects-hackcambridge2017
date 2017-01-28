import pandas as pd
import numpy as np
from data_cleaning import *
from feature_scaling import *

df = pd.DataFrame.from_csv('../DataFiles/train.csv')

X = df['Attr1'].values
nans = np.isnan(X)
print('nans before: ',np.sum(nans))

df = clean_data(df)
X = df['Attr1'].values
nans = np.isnan(X)
print('nans after: ',np.sum(nans))

print('Before scaling:')
print('minmax: (',min(df['Attr1'].values),',',max(df['Attr1'].values),')')
print('mean:',np.mean(df['Attr1'].values))
print('std: ',np.std(df['Attr1'].values))
df = scale_features(df)
print('After scaling:')
print('(',min(df['Attr1'].values),',',max(df['Attr1'].values),')')
print('mean: ',np.mean(df['Attr1'].values))
print('max: ',np.std(df['Attr1'].values))
