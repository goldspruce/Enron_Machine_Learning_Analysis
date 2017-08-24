

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
''
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
# print each
labels, features = targetFeatureSplit(data)



knn =  KNeighborsClassifier()
scaler = MinMaxScaler()
sss = StratifiedShuffleSplit(n_splits = 100, random_state=42)

pipe =  Pipeline(steps=[('scaling',scaler), ("knn", knn)])

param_grid = {"knn__algo": ["auto", "ball_tree", "kd_tree", "brute"],
              "knn__leaf_size": [1, 2, 4, 6, 10],
              }

dtcclf = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
dtcclf.fit(features, labels)
bestclf = dtcclf.best_estimator_
print "Best KNN"
print bestclf