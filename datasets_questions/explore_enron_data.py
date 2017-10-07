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


'''
counter2 =0
for key in enron_data:
    if enron_data[key]['poi']==1:
        counter2 +=1
print "Total Poi number is: ", counter2
'''

#What is the total value of the stock belonging to James Prentice?
print enron_data['PRENTICE JAMES']['total_stock_value']

#How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

##What is the value of stock options exercised by Jeffrey K Skilling?
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

#How many folks in this dataset have a quantified salary? What about a known email address?
salary_counter = 0
email_counter = 0
for key in enron_data:
    if enron_data[key]['salary'] != 'NaN':
        salary_counter +=1
    if enron_data[key]['email_address'] !='NaN':
        email_counter +=1

print "salary counter is:" , salary_counter
print "email counter is:" , email_counter


#How many people in the E+F dataset (as it currently exists) have 'NaN' for their total payments? What percentage of people in the dataset as a whole is this?
total_payment_counter =0
for key in enron_data:
    if enron_data[key]['total_payments'] == 'NaN':
        total_payment_counter +=1

print "total payment missing percentage:" , float(total_payment_counter)/len(enron_data)
print "total payment missing number: " ,total_payment_counter
#How many POIs in the E+F dataset have 'NaN' for their total payments? What percentage of POI's as a whole is this?
total_payment_poi_counter =0
poi_counter = 0
for key in enron_data:
    if enron_data[key]['poi'] == 1:
        poi_counter += 1
        if enron_data[key]['total_payments'] == 'NaN':
            total_payment_poi_counter += 1.0

print "poi NaNs: ", total_payment_poi_counter
print "number of pois", poi_counter
print "poi NaN percentage: ", total_payment_poi_counter/poi_counter
