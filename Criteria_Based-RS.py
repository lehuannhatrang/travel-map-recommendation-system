import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
from scipy import sparse 
import sys

criteria = list(map(int, sys.argv[1:]))

df = pd.read_csv(open("/home/yntn/Thesis/travel-map-recommendation-system/data/foody-hcm-rating-restaurant.csv",'r'))
go = df.values[:,-5:]
vec = np.array([criteria]*len(go))
sim = paired_distances(vec,go)
result = np.where(sim == np.amin(sim))
print (result[0])