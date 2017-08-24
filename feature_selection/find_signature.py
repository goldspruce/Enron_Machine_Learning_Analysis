#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
import scipy

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.

words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"

word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )
print authors.count(0)
print authors.count(1)

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


#print len(features_train)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

print len(vectorizer.get_feature_names())

#print scipy.sparse.csr_matrix.getnnz(features_train)


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

print features_train.shape

### your code goes here

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = DecisionTreeClassifier()#random_state=0)
    clf.fit(features_train, labels_train)
    
    return clf
    
clf = classify(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)
print acc

fi=[]
n=-1
for each in clf.feature_importances_:
	n+=1
	if each > 0.2:
		print n
		index=n
		fi.append(each)

print fi
print vectorizer.get_feature_names()[index]


