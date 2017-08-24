#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###


#########################################################

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train,mss):
    clf = DecisionTreeClassifier(random_state=0,min_samples_split=mss)
    clf.fit(features_train, labels_train)
    return clf

def classit(n):
    return classify(features_train, labels_train, n)
        
#clf50 = classit(50)
#clf2 = classit(2)

def pred(n):
    return classit(n).predict(features_test)
    
#pred50 = classit(50).predict(features_test)
#pred2 = classit(2).predict(features_test)

acc_min_samples_split_50 = accuracy_score(pred(50),labels_test)
acc_min_samples_split_2 = accuracy_score(pred(2),labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

print submitAccuracies()
