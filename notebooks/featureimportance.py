import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = pd.DataFrame.from_csv('../DataFiles/train.csv')
data = pd.DataFrame.fillna(data, value=0)

cols = [col for col in data.columns if col != "class"]
features = data[cols]
bankrupt = data['class']

x_train, x_test, y_train, y_test = train_test_split(
    features, bankrupt, test_size=0.8, random_state=1)

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators=100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x_train, y_train)
# Take the same decision trees and run it on the test data
output = forest.predict(x_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]
indexcols = [features.columns[i] for i in indices]
print(indexcols)

attrs = {}
with open("../DataFiles/column_description.csv") as csvfile:
    for line in csvfile:
        key, value = line.split(",")
        attrs[key.strip()] = value.strip()


for indexcol in indexcols:
    print(attrs[indexcol])
