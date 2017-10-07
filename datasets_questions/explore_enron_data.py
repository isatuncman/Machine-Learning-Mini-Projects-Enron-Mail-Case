#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# # of people in enron dataset
print len(enron_data)

# Invesitagete a sample person and his features
counter = 0
for key in enron_data:
    print key
    print len(enron_data[key])
    print enron_data[key]

    counter +=1
    if counter == 1:
        break


counter2 =0
for key in enron_data:
    if enron_data[key]['poi']==1:
        counter2 +=1
print "Total Poi number is: ", counter2
