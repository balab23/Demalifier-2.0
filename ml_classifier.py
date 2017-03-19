import numpy as np
import urllib
import math
import random
import time
from sklearn import preprocessing
from collections import defaultdict
from fractions import Fraction 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import pandas

spam=pandas.read_csv('final.csv',delimiter=' ').as_matrix()

X_train=spam[3:,:-1]
Y_train=spam[3:,-1]

X_test=list(spam[2,:-1])
print X_test

classifier=OneVsRestClassifier(LinearSVC())
classifier.fit(X_train,Y_train)
print classifier.predict([X_test])



