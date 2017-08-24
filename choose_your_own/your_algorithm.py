#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

print 1

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

print 2

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
print 3


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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

grade_fast = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==0]
bumpy_fast = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==0]
grade_slow = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==1]
bumpy_slow = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==1]


#### test visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

try:
    prettyPicture(pred_knn(5), features_test, labels_test)
except NameError:
    pass