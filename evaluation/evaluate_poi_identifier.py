#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

newclf = tree.DecisionTreeClassifier()
newclf.fit(X_train, y_train)
y_pred = newclf.predict(X_test)

print newclf.score(X_test, y_test)

counter = 0
for i in y_test:
    if i >0:
        counter += 1
print counter

print recall_score(y_test, y_pred)
print precision_score(y_test, y_pred)
