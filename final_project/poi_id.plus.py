#!/usr/bin/python

import sys
import pickle

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import grid_search 
from time import time

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_stock_value','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi','ratio_exercised']
# You will need to use more features, 8 existing features, 3 new ones I created

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0) # remove TOTAL entry, not a person, but a total of everything

### Task 2: Remove outliers
### remove NaN and outliers (4 largest salaries, LAy, Skilling, Fastow, Frevert)
#how to define outliers
'''
print len(data_dict)
to_remove = []
n=0
for key in data_dict:
    if data_dict[key]['salary'] == 'NaN':
    	to_remove.append(key)
    elif data_dict[key]['salary']>800000:
    	print key
    	print data_dict[key]['salary']
    	to_remove.append(key)
    	n+=1
print n
for each in to_remove:
	data_dict.pop(key)
print len(data_dict)
'''

### Task 3: Create new feature(s)
# 3 ratio_exercised, ratio_to_poi, ratio_from_poi

def new_feature(numerator,denominator):
    new_feature=[]

    for i in data_dict:
        if data_dict[i][numerator]=="NaN" or data_dict[i][denominator]=="NaN":
            new_feature.append("NaN")
        elif data_dict[i][numerator]<0 or data_dict[i][denominator]<=0:
            new_feature.append("NaN")
        else:
            new_feature.append(float(data_dict[i][numerator]) / data_dict[i][denominator])
    return new_feature

### create two lists of new features
ratio_exercised=new_feature("exercised_stock_options","total_stock_value")
ratio_to_poi=new_feature("from_poi_to_this_person","to_messages")
ratio_from_poi=new_feature("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["ratio_exercised"] = ratio_exercised[count]
    data_dict[i]["ratio_to_poi"] = ratio_to_poi[count]
    data_dict[i]["ratio_from_poi"] = ratio_from_poi[count]
    count +=1

### store to my_dataset for easy export below
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
#for each in data:
#	print each
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### plot new features

n=0
for point in data:
    ratio_exercised = point[8]
    plt.scatter( n, ratio_exercised )
    n+=1
    #add smooth
    #if point[0] == 1:
     #   plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("ratio exercised")
#plt.show()

n=0
for point in data:
    to_poi  = point[9]
    plt.scatter( n, to_poi )
    n+=1
    #add smooth
    #if point[0] == 1:
     #   plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("ratio of all emails sent to POI")
#plt.show()

n=0
for point in data:
    from_poi = point[10]
    plt.scatter( n, from_poi )
    n+=1
    #add smooth
    #if point[0] == 1:
     #   plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("ratio of all emails received from POI")
#plt.show()

# Provided to give you a starting point. Try a variety of classifiers.

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

def classify_NB(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
    
clf = classify_NB(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(pred,labels_test)
print precision_score(pred,labels_test)
print recall_score(pred,labels_test)
print f1_score(pred,labels_test)

def classify_DTC(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    return clf
    
clf = classify_DTC(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(pred,labels_test)
print precision_score(pred,labels_test)
print recall_score(pred,labels_test)
print f1_score(pred,labels_test)

def classify_knn(features_train,labels_train,n):
    neigh = KNeighborsClassifier(n)
    neigh.fit(features_train, labels_train)
    return neigh

def classit_knn(n):
    return classify_knn(features_train, labels_train, n)

def pred_knn(n):
    return classit_knn(n).predict(features_test)

pred_5 = accuracy_score(pred_knn(5),labels_test)
pred_8 = accuracy_score(pred_knn(8),labels_test)
pred_20 = accuracy_score(pred_knn(20),labels_test)

def submitAccuracies():
  return {"5":round(pred_5,3),
          "8":round(pred_8,3),
          "20":round(pred_20,3)}
print submitAccuracies()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


dtree =  DecisionTreeClassifier()
scaler = MinMaxScaler()
sss = StratifiedShuffleSplit(n_splits = 100, random_state=42)


gs =  Pipeline(steps=[('scaling',scaler), ("dt", dtree)])

param_grid = {"dt__criterion": ["gini", "entropy"],
              "dt__min_samples_split": [2, 4, 6],
              "dt__max_depth": [None, 2, 4, 6],
              "dt__min_samples_leaf": [1, 2, 4, 6],
              "dt__max_leaf_nodes": [None, 2, 4, 6],
              }


dtcclf = GridSearchCV(gs, param_grid, scoring='f1', cv=sss)

dtcclf.fit(features, labels)

clf = dtcclf.best_estimator_

print clf

best = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')


best.fit(features_train, labels_train)
pred = best.predict(features_test)
print accuracy_score(pred,labels_test)
print precision_score(pred,labels_test)
print recall_score(pred,labels_test)
print f1_score(pred,labels_test)

'''

sss = StratifiedShuffleSplit(n_splits = 100, random_state=42)

scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)


for i_train, i_test in sss.split(features, labels):
	features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
	labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]
	# clf = GaussianNB()
	clf = classify_NB(features_train, labels_train)
	pred = clf.predict(features_test)
	print accuracy_score(pred,labels_test)
	print precision_score(pred,labels_test)
	print recall_score(pred,labels_test)
	print f1_score(pred,labels_test)
	# DTC
	clf = classify_DTC(features_train, labels_train)
	pred = clf.predict(features_test)
	print accuracy_score(pred,labels_test)
	print precision_score(pred,labels_test)	
	print recall_score(pred,labels_test)
	print f1_score(pred,labels_test)
	#KMM
	pred_5 = accuracy_score(pred_knn(5),labels_test)
	pred_8 = accuracy_score(pred_knn(8),labels_test)
	pred_20 = accuracy_score(pred_knn(20),labels_test)
	print submitAccuracies()

'''
'''


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

 # my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl). 
 '''