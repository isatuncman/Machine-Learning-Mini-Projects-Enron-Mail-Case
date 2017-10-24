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

email_features = ['to_messages', 'from_poi_to_this_person',
 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

POI_label = ['poi']

features_list = ['poi','salary'] # You will need to use more features

all_features = POI_label + financial_features + email_features


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

#Missing fields:  {'salary': 51, 'to_messages': 60, 'deferral_payments': 107,
# 'total_payments': 21, 'long_term_incentive': 80, 'loan_advances': 142,
#'bonus': 64, 'restricted_stock': 36, 'restricted_stock_deferred': 128,
#'total_stock_value': 20, 'shared_receipt_with_poi': 60, 'from_poi_to_this_person': 60,
#'exercised_stock_options': 44, 'from_messages': 60, 'other': 53,
#'from_this_person_to_poi': 60, 'deferred_income': 97, 'expenses': 51,
#'email_address': 35, 'director_fees': 129}



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
    if  data_dict[key]['from_this_person_to_poi'] != 'NaN':
        data_dict[key]['from_this_person_fraction'] = float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])

    else:
        data_dict[key]['from_this_person_fraction'] = 'NaN'

    if data_dict[key]['from_poi_to_this_person'] != 'NaN':
        data_dict[key]['from_poi_to_this_person_fraction'] = float(data_dict[key]['from_poi_to_this_person']) / float(data_dict[key]['to_messages'])
    else:
        data_dict[key]['from_poi_to_this_person_fraction'] = 'NaN'
    if data_dict[key]['salary'] != 'NaN' and data_dict[key]['bonus'] != 'NaN':
        data_dict[key]['salary_bonus_ratio'] = float(data_dict[key]['salary'])/float(data_dict[key]['bonus'])
    else:
        data_dict[key]['salary_bonus_ratio'] = 'NaN'


    #print 'Bonus_salary_ratio', data_dict[key]['bonus_salary_ratio']
    #print 'from_this_person_fraction', data_dict[key]['from_this_person_fraction']
    #print 'from_poi_to_this_person_fraction', data_dict[key]['from_poi_to_this_person_fraction']

### Store to my_dataset for easy export below.
my_dataset = data_dict

#for key in data_dict:
#    print key, data_dict[key]['bonus_salary_ratio'], data_dict[key]['salary']
#    print key, data_dict[key]['from_this_person_fraction'], data_dict[key]['from_messages']
#    print key, data_dict[key]['from_poi_to_this_person_fraction'], data_dict[key]['to_messages']

all_features = all_features + ['from_this_person_fraction'] + ['from_poi_to_this_person_fraction'] + ['salary_bonus_ratio']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#print 'Labels:', labels
#print 'Features:', features



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Feature selection algorithm
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k = 22)

#Scaler function
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

#Import pipeline
from sklearn.pipeline import Pipeline



# Provided to give you a starting point. Try a variety of classifiers.
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
pipe_g = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])
parameters_g ={'SKB__k': range(1,23)}


#Decision Tree
from sklearn import tree
pipe_dt = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("DTC", tree.DecisionTreeClassifier())])
parameters_dt = {'SKB__k': [1,2,3,4,5,10,23],
'DTC__criterion': ['gini', 'entropy'],
'DTC__min_samples_split': [2, 10, 20],
'DTC__max_depth': [None, 2, 5, 10],
'DTC__min_samples_leaf': [1, 5, 10],
'DTC__max_leaf_nodes': [None, 5, 10, 20]}

#K neighbors
from sklearn.neighbors import KNeighborsClassifier
pipe_kn = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("KNN", KNeighborsClassifier())])
parameters_kn ={"SKB__k": range(1,10),
    "KNN__n_neighbors": [2,3,4,5,6,8,10],
    "KNN__weights": ["uniform", "distance"],
    "KNN__algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}





### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#   train_test_split(features, labels, test_size=0.3, random_state=42)

#Due to small dataset, we use tratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, cross_val_score
sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)

#Use GridSearchCV to find best parameters
from sklearn.model_selection import GridSearchCV



#Naive Bayes GridsearchCV
gs = GridSearchCV(pipe_kn, param_grid = parameters_kn, cv=sk_fold, scoring = 'f1')
gs.fit(features, labels)
clf = gs.best_estimator_


print 'best algorithm using strat_s_split'
print clf
print gs.best_params_
print gs.best_score_

skb_step = gs.best_estimator_.named_steps['SKB']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in skb_step.scores_ ]

# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.4f' % elem for elem in  skb_step.pvalues_ ]

# Get SelectKBest feature names, whose indices are stored in 'skb_step.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(all_features[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in skb_step.get_support(indices=True)]

print 'Selected features:', features_selected_tuple
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, all_features)
