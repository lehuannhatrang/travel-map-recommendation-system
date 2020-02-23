import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
from scipy import sparse 
import sys

criteria = list(map(int, sys.argv[1:]))

df = pd.read_csv(open("/home/yntn/Thesis/data/restaurant-foody-ver2.csv",'r'))
go = df.values[:,-5:]
vec = np.array([criteria]*len(go))
sim = paired_distances(vec,go)
result = np.argsort(sim)[:15]
print (result)