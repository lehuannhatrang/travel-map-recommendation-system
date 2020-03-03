import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
from scipy import sparse 
import sys
from functools import reduce 

criteria = list(map(float, sys.argv[1:]))

df = pd.read_csv(open("/home/yntn/Thesis/data/restaurant-foody-ver2.csv",'r'))
go = df.values[:,-5:]
vec = np.array([criteria]*len(go))
sim = paired_distances(vec,go)


min_id = np.where(sim == np.amin(sim))[0]

if (len(min_id) > 15):
    import secrets
    secure_radom = secrets.SystemRandom()
    result = secure_radom.sample(set(min_id), 15)
else:
    result = np.argsort(sim)[:15]

res = ''
for i in range(len(result)-1):
    res += str(result[i]) + ','

res += str(result[len(result)-1])

print(res)
   


