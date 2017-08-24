#!/usr/bin/python

### See .ipynb file for documentation ###
###

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_stock_value','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi','ratio_exercised']
# You will need to use more features, 8 existing features, 3 new ones I created

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

import pprint

features_list = ['poi','salary','total_stock_value','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi','ratio_exercised']

print len(data_dict)
print len(data_dict['ALLEN PHILLIP K'])

data_dict.pop('TOTAL', 0)
data_dict.pop('TRAVEL AGENCY IN THE PAR', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
print "Three entries popped"
len(data_dict)

n=0
for each in data_dict:
    if data_dict[each]['poi']:
        n+=1
print n

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

### create lists of new features
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

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

n=0
for point in data:
    ratio_exercised = point[8]
    plt.scatter( n, ratio_exercised )
    n+=1
plt.xlabel("ratio exercised")
plt.show()

n=0
for point in data:
    ratio_to_poi  = point[9]
    if ratio_to_poi == 1.0:
        print point
    plt.scatter( n, ratio_to_poi )
    n+=1
plt.xlabel("ratio of all emails sent to POI")
plt.show()

n=0
for point in data:
    ratio_from_poi  = point[10]
    plt.scatter( n, ratio_from_poi )
    n+=1
plt.xlabel("ratio of all from POI")
plt.show()

K_best = SelectKBest(k=5)
# Use that instance to extract the best features:
features_kbest = K_best.fit_transform(features, labels)
print "Shape of features after applying SelectKBest -> ", features_kbest.shape
print data_dict["ALLEN PHILLIP K"]
print features_kbest[0]
print features[0]
print data_dict["BANNANTINE JAMES M"]
print features_kbest[2]
print features[2]

features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

def classify_NB(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

def class_run(features_train, labels_train, features_test, labels_test):
    clf = classify_NB(features_train, labels_train)
    pred = clf.predict(features_test)
    return {"Accuracy":accuracy_score(pred,labels_test),\
                          "Precision":precision_score(pred,labels_test),\
                          "Recall":recall_score(pred,labels_test),\
                          "F1":f1_score(pred,labels_test)}

def print_it(title,input):
    print title
    answer = input
    for each in answer:
        print each," : ",answer[each]
    print


features_old = ['poi','salary','total_stock_value','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi','ratio_exercised']
features_new = ['poi','ratio_to_poi','ratio_from_poi','ratio_exercised']
features_first = ['poi','salary','total_stock_value','exercised_stock_options']
features_second = ['poi','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi','ratio_exercised']
features_F = ['poi','salary','total_stock_value','exercised_stock_options','ratio_exercised']
features_E = ['poi','from_poi_to_this_person','from_this_person_to_poi','from_messages','to_messages','ratio_to_poi','ratio_from_poi']

print "all 11 features"
print
print_it("NB: 0.1 test size",class_run(features_train1, labels_train1, features_test1, labels_test1))
print_it("NB: 0.3 test size",class_run(features_train3, labels_train3, features_test3, labels_test3))
print_it("NB: 0.5 test size",class_run(features_train5, labels_train5, features_test5, labels_test5))

data = featureFormat(my_dataset, features_old, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

print "features_old = the original 8"
print
print_it("NB: 0.1 test size",class_run(features_train1, labels_train1, features_test1, labels_test1))
print_it("NB: 0.3 test size",class_run(features_train3, labels_train3, features_test3, labels_test3))
print_it("NB: 0.5 test size",class_run(features_train5, labels_train5, features_test5, labels_test5))

data = featureFormat(my_dataset, features_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

print "features_new = 3 new features"
print
print_it("NB: 0.1 test size",class_run(features_train1, labels_train1, features_test1, labels_test1))
print_it("NB: 0.3 test size",class_run(features_train3, labels_train3, features_test3, labels_test3))
print_it("NB: 0.5 test size",class_run(features_train5, labels_train5, features_test5, labels_test5))

data = featureFormat(my_dataset, features_first, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

print "features_first = poi + 1,2,3   ($ related)"
print
print_it("NB: 0.1 test size",class_run(features_train1, labels_train1, features_test1, labels_test1))
print_it("NB: 0.3 test size",class_run(features_train3, labels_train3, features_test3, labels_test3))
print_it("NB: 0.5 test size",class_run(features_train5, labels_train5, features_test5, labels_test5))

data = featureFormat(my_dataset, features_second, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

print "poi + 4,5,6,7 (email related)"
print
print_it("NB: 0.1 test size",class_run(features_train1, labels_train1, features_test1, labels_test1))
print_it("NB: 0.3 test size",class_run(features_train3, labels_train3, features_test3, labels_test3))
print_it("NB: 0.5 test size",class_run(features_train5, labels_train5, features_test5, labels_test5))

def classify_DTC(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    return clf

def class_run_DTC(features_train, labels_train, features_test, labels_test):
    clf = classify_DTC(features_train, labels_train)
    pred = clf.predict(features_test)
    return {"Accuracy":accuracy_score(pred,labels_test),\
                          "Precision":precision_score(pred,labels_test),\
                          "Recall":recall_score(pred,labels_test),\
                          "F1":f1_score(pred,labels_test)}

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
features_train5, features_test5, labels_train5, labels_test5 = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

print "All 11 features, DTC"
print
print_it("DTC: 0.1 test size",class_run_DTC(features_train1, labels_train1, features_test1, labels_test1))
print_it("DTC: 0.3 test size",class_run_DTC(features_train3, labels_train3, features_test3, labels_test3))
print_it("DTC: 0.5 test size",class_run_DTC(features_train5, labels_train5, features_test5, labels_test5))

def classify_KNN(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = KNeighborsClassifier(3)
    clf.fit(features_train, labels_train)
    return clf

def class_run_KNN(features_train, labels_train, features_test, labels_test):
    clf = classify_KNN(features_train, labels_train)
    pred = clf.predict(features_test)
    return {"Accuracy":accuracy_score(pred,labels_test),\
                          "Precision":precision_score(pred,labels_test),\
                          "Recall":recall_score(pred,labels_test),\
                          "F1":f1_score(pred,labels_test)}

print "All 11 features, KNN, k=5"
print
print_it("KNN: 0.1 test size",class_run_KNN(features_train1, labels_train1, features_test1, labels_test1))
print_it("KNN: 0.3 test size",class_run_KNN(features_train3, labels_train3, features_test3, labels_test3))
print_it("KNN: 0.5 test size",class_run_KNN(features_train5, labels_train5, features_test5, labels_test5))

knn =  KNeighborsClassifier()
scaler = MinMaxScaler()
sss = StratifiedShuffleSplit(n_splits = 100, test_size=0.1,random_state=42)

pipe = Pipeline(steps=[('scaling', scaler), ("knn", knn)])

param_grid = {"knn__n_neighbors": [1, 3, 5, 8, 10, 12, 14],
              "knn__algorithm": ["auto","ball_tree", "kd_tree", "brute"],
              "knn__leaf_size": range(3,10,1),
              "knn__p": [1,2]
             }

knnclf = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
knnclf.fit(features, labels)
bestknn = knnclf.best_estimator_
bestparams = knnclf.best_params_
print "Best KNN"
print bestknn
print bestparams

best = KNeighborsClassifier(algorithm='auto', leaf_size=3, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')
best.fit(features_train1, labels_train1)
pred = best.predict(features_test1)
print "KNN grid: accuracy, precision, recall, F1"
print accuracy_score(pred,labels_test1)
print precision_score(pred,labels_test1)
print recall_score(pred,labels_test1)
print f1_score(pred,labels_test1)

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)
        
dump_classifier_and_data(bestknn, data_dict, features_list)

with open("my_classifier.pkl", "r") as file:
    clf = pickle.load(file)
print clf

with open("my_dataset.pkl", "r") as file:
    data = pickle.load(file)
for person in data:
    print person, data[person]
    break

with open("my_feature_list.pkl", "r") as file:
    features = pickle.load(file)
print features