import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from math import *

sys.path.append('../preprocessing')
from data_cleaning import *
from feature_scaling import *
from liamfeatureimportance import *

def bayesian( output, y_test ):
    errors = np.where(y_test!=output,y_test,output)
    true_positives  = np.where(np.logical_and(y_test,output))
    true_negatives  = np.where(np.logical_not(np.logical_or(y_test,output)))
    false_positives = np.where(y_test<output)
    false_negatives = np.where(y_test>output)

    n_true_positives = len(true_positives[0])
    n_true_negatives = len(true_negatives[0])
    n_false_negatives = len(false_negatives[0])
    n_false_positives = len(false_positives[0])

    n_sum = float(n_true_positives+n_false_positives+n_false_negatives+n_true_negatives)
    n_bank = float(n_true_positives+n_false_negatives)
    n_pred = float(n_true_positives+n_false_positives)
    P_pred_bank = float(n_true_positives) / n_bank
    P_bank = n_bank/n_sum
    P_predicted = n_pred/n_sum

    print('sum: ', n_sum)
    print('true positives: ' , n_true_positives)
    print('true negatives: ' , n_true_negatives)
    print('false positives: ' , n_false_positives)
    print('false negatives: ' , n_false_negatives)
    print('P(predicted|bankrupt): ', P_pred_bank)
    return P_pred_bank, true_positives, false_negatives



data = pd.DataFrame.from_csv('../DataFiles/train.csv')
#drop_indices = []
#print( 'dropping years <4')
#for idx in data.index:
#    if (data['year'].values[idx]<4):
#        drop_indices = drop_indices + [idx]
#data = data.drop(drop_indices)
#print( 'remaining samples: ', len(data.index))


# manual data importance stuff
#data = data[['Attr27_Nans','Attr34','Attr46','Attr27','Attr58','Attr56','Attr40','Attr42','Attr41','Attr22','Attr9','class']]

cols = [col for col in data.columns if col != "class"]
features = data[cols]
bankrupt = data['class']

x_train = features
y_train = bankrupt
#x_train, x_test, y_train, y_test = train_test_split(
#    features, bankrupt, test_size=0.0)

testdata = pd.DataFrame.from_csv('../DataFiles/test.csv')
cols = [col for col in testdata.columns if col != "class"]
features = testdata[cols]
bankrupt = testdata['class']

x_test = features
y_test = bankrupt

x_train = clean_data(x_train)
x_test  = clean_data(x_test)
#x_train = scale_features(x_train)
#x_test  = scale_features(x_test)

# Create the random forest object which will include all the parameters
# for the fit
cls = GradientBoostingClassifier(n_estimators=100,     # Number of trees in ensemble
                                 learning_rate=0.5,
#                                criterion='gini', # Decision trees maximise information gain
           #                     n_jobs=8,            # Go faster number
                                random_state=1)      # Set RNGesus
# smaller version
#glade = RandomForestClassifier(n_estimators=10,     # Number of trees in ensemble
#                               criterion='entropy', # Decision trees maximise information gain
#                               n_jobs=8,            # Go faster number
#                               random_state=1)      # Set RNGesus


print('Initial training:')
# Fit the training data to the Survived labels and create the decision trees
cls = cls.fit(x_train, y_train)
# Take the same decision trees and run it on the test data
print('Score: ',cls.score(x_test, y_test))
output = cls.predict(x_test)
proba = cls.predict_proba(x_test)

f = open('answers.txt','w')
for idx in range(len(output)):
    f.write(str(output[idx])+'\n')

PAB, true_positives, false_negatives = bayesian(output,y_test)
print('Mean prob of true positives:')
print( np.mean(proba[true_positives,1]))
print('std prob of true positives:')
print( np.std(proba[true_positives,1]))
print('Mean prob of false negatives:')
print( np.mean(proba[false_negatives,0]))
print('Std prob of false negatives:')
print( np.std(proba[false_negatives,0]))

#
importances = feature_importance(features,cls)
print('Importances:')
print(importances)
#
#print('Iterating:')
#maxPAB = PAB
#max_importance = len(importances)
#jump = floor(max_importance/2)
#jump = 20
#for iteration in range(10):
#    max_importance = max_importance - jump
#    print('Max importance: ', max_importance)
#    if( max_importance <= 0 ):
#        break
#    newdata = data[importances[:max_importance]+['class']]
#    cols = [col for col in newdata.columns if col != "class"]
#    features = newdata[cols]
#    bankrupt = newdata['class']
#
#    x_train, x_test, y_train, y_test = train_test_split(
#        features, bankrupt, test_size=0.2, random_state=1)
#
#    x_train = clean_data(x_train)
#    x_test  = clean_data(x_test)
#    print('Iter ', iteration+1, ':')
#    # Fit the training data to the Survived labels and create the decision trees
#    cls = cls.fit(x_train, y_train)
#    # Take the same decision trees and run it on the test data
#    print('Score: ',cls.score(x_test, y_test))
#    output = cls.predict(x_test)
#    proba = cls.predict_proba(x_test)
#    PAB, true_positives, false_negatives = bayesian(output,y_test)
#    if ( PAB > maxPAB ):
#        #jump /= -2
#    #else:
#        maxPAB = PAB
#        #jump /= 2
#    #jump = int(jump)
#
#print('max PAB: ', maxPAB)
