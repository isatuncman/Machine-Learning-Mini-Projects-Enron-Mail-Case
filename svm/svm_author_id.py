#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC




### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
#small training set
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#clf = SVC(kernel = 'linear')
clf = SVC(kernel = 'rbf', C = 10000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)

#Find the number of class 1 predictions
result = sum(pred[pred==1])
print(result)

'''
counter = 0
for i in range(len(pred)):
    if pred[i] == labels_test[i]:
        counter += 1.0

accuracy = counter/ len(pred)
print accuracy
'''

#########################################################
