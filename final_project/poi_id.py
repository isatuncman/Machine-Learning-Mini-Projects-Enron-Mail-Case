#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from matplotlib import pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments',
'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person',
 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

POI_label = ['poi']

features_list = ['poi','salary'] # You will need to use more features



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# How many data points (people) are in the dataset?
print 'Number of employees: ', len(data_dict)


# For each person, how many features are available?
for key in data_dict:
    print 'features in dataset:', data_dict[key]
    print 'Number of features for each employee: ', len(data_dict[key])
    break

#How many POIs are there in the E+F dataset?
poi_counter = 0
for key in data_dict:
    if data_dict[key]['poi'] == True:
        poi_counter +=1
print "Number of poi's in dataset: ", poi_counter

#Explore all missing data
missing_dict ={}
for key in data_dict:
    for feature in data_dict[key]:
        if data_dict[key][feature] == 'NaN':
            if missing_dict.get(feature, 0) > 0:
                missing_dict[feature] += 1
            else:
                missing_dict[feature] =1
print 'Missing fields: ', missing_dict




### Task 2: Remove outliers
features = ["salary", "bonus"]


#for key in data_dict:
#    print key


# Plot before outliers are removed
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

# Plot after outliers are removed
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Task 3: Create new feature(s)

#1) bonus/salary ratio: Can be useful if some low salaried people takes
#large amount of bonuses

#2)  from_this_person_to_poi' / 'from_messages' ratio : the fraction of messages
#from this person to poi

#3) 'from_poi_to_this_person' / 'to_messages'the fraction of messages
#to this person from poi

for key in data_dict:
    data_dict[key]['bonus_salary_ratio'] = round((float(data_dict[key]['bonus']) / float(data_dict[key]['salary'])),3)
    data_dict[key]['from_this_person_fraction'] = round((float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])),3)
    data_dict[key]['from_poi_to_this_person_fraction'] = round((float(data_dict[key]['from_poi_to_this_person']) / float(data_dict[key]['to_messages'])),3)

    #print 'Bonus_salary_ratio', data_dict[key]['bonus_salary_ratio']
    #print 'from_this_person_fraction', data_dict[key]['from_this_person_fraction']
    #print 'from_poi_to_this_person_fraction', data_dict[key]['from_poi_to_this_person_fraction']

### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
