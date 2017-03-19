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

spam=pandas.read_csv('spam.csv',delimiter=' ').as_matrix()
phish=pandas.read_csv('phish.csv', delimiter=' ').as_matrix()
mal=pandas.read_csv('mal.csv', delimiter=' ').as_matrix()

spam=list(spam)
phish=list(phish)
mal=list(phish)
final=[]

#np.append(spam,np.asarray([1 for i in range(spam.shape[0])]),axis=1)
#print np.asarray([1 for i in range(spam.shape[0])]).shape[0]
for i, sp in enumerate(spam):
	spam[i]=list(spam[i])
	spam[i].append(1)
	final.append(spam[i])
	print spam[i]
	print spam[i][-1]
	print spam[i][-2]

for i, sp in enumerate(phish):
	phish[i]=list(phish[i])
	phish[i].append(2)
	final.append(phish[i])

for i, sp in enumerate(mal):
	mal[i]=list(mal[i])
	mal[i].append(3)
	final.append(mal[i])

final=np.asarray(final)
#print final
np.savetxt('final.csv',np.asarray(final))




