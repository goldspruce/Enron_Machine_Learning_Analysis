sklearn.feature_extraction.text.CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
b = vectorizer.fit()
b = vectorizer.transform()
vectorizer.vocabulary_.get('great')

from nltk.corpus import stopwords
sw = stopwords.words('english')
import nltk
nltk.download()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
stemmer.stem('unresponsive')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")vectorizer.fit_transform(word_data)

rom sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(features_train, labels_train)
    
    return clf
Quiz: [Object Object]
accuracy = .908
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData