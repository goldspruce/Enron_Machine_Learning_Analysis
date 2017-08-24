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
import pandas as pd
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)
print len(enron_data["SKILLING JEFFREY K"])
count=0
for each in enron_data:
	print each
	if enron_data[each]['poi']:
		count+=1
print count	
list_new = pd.read_csv('../final_project/poi_names.txt', skiprows=2, header=None)
print list_new
print len(list_new)

print enron_data["PRENTICE JAMES"]["total_stock_value"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]

print enron_data["SKILLING JEFFREY K"]["total_payments"]

count=0
for each in enron_data:
	if enron_data[each]['email_address']!='NaN':
		count+=1
print count

count=0
for each in enron_data:
	if enron_data[each]['salary']!='NaN':
		count+=1
print count

count=0
for each in enron_data:
	if enron_data[each]['total_payments']=='NaN':
		count+=1
print count

print count * 1.0 / len(enron_data) 

count=0
y=0
for each in enron_data:
	if enron_data[each]['poi']:
		count+=1
		if enron_data[each]['total_payments']=='NaN':
			y+=1
print y
print "poi",count

print y * 100 / count
print 10 * 100 / 28