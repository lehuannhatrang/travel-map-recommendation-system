import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *

import sys

criteria = list(map(float, sys.argv[1:]))

df = pd.read_json(open("../Data/foody-hcm-rating-restaurant_official.json",'r'))
dt = df.values
go = dt[:,-5:]
vec = np.array([criteria]*len(go))
sim = paired_distances(vec,go)


min_id = np.where(sim == np.amin(sim))[0]

if (len(min_id) > 100):
    result = np.random.choice(min_id, 100, replace = False)
else:
    result = np.argsort(sim)[:100]

res = ''
for i in range(len(result)-1):
    res += str(int(dt[result[i]][0])) + ','

res += str(int(dt[result[len(result)-1]][0]))

print(res)
   


