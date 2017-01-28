import sklearn
import pandas as pd

data = pd.DataFrame.from_csv('../DataFiles/train.csv')

# generate a DataFrame of bools saying if there are NaNs and add it on
missing = data.drop("class", axis=1).drop("year", axis=1).isnull()
missing.rename(columns={x: x+"_Nans" for x in missing.columns}, inplace=True)
data_withNaNs = pd.concat([data, missing], axis='col')

# generate a DataFrame of the total NaNs per index and add it on
NaN_total = pd.DataFrame({"Total_NaNs":missing.sum(axis=1)})
data_withNaNs = pd.concat([data_withNaNs,NaN_total], axis='col')

# output the resulting 131 columns DataFrame as a csv
data_withNaNs.to_csv("../DataFiles/train_withNaNinfo.csv")
