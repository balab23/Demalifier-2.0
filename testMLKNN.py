import numpy as np
import urllib
import math
import random
import time
import pandas

from sklearn import preprocessing
from collections import defaultdict
from fractions import Fraction 

def dissimilarityMeasure(X, Y): 
	""" Simple matching disimilarity measure """
	return np.sum(X!=Y, axis = 1)

# df = pandas.read_csv("centroids.csv", delimiter= ' ').as_matrix()
def ML_KNN(feature):
	df = np.genfromtxt("centroids.csv", delimiter= ' ')

	labels = []
	#feature = np.random.rand(1,8)
	for x in xrange(len(df)):
		t = dissimilarityMeasure(feature, df[x])
		if t < 5:
			labels.append(x)
	return labels




