import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.append('../preprocessing')
from data_cleaning import *
from feature_scaling import *


data = pd.DataFrame.from_csv('../DataFiles/train.csv')

data = clean_data(data)
data = scale_features(data)

cols = [col for col in data.columns if col != "class"]
features = data[cols]
bankrupt = data['class']

x_train, x_test, y_train, y_test = train_test_split(
    features, bankrupt, test_size=0.2, random_state=1)

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators=100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x_train, y_train)
# Take the same decision trees and run it on the test data
output = forest.predict(x_test)
print(forest.score(x_test, y_test))
