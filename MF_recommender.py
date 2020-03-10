
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
from scipy import sparse 
import sys
import os
import pickle

userId = list(map(int, sys.argv[1:]))[0]

path = '/home/yntn/Thesis/travel-map-recommendation-system/'


if (os.path.exists(path + 'MF_pred_for_all_user-new')):
    pred_for_all_user = pickle.load(open(path + 'MF_pred_for_all_user-new','rb'))
elif (os.path.exists(path + 'MF_pred_for_all_user-old')):
    pred_for_all_user = pickle.load(open(path + 'MF_pred_for_all_user-old','rb'))
else:
    raise NameError("Missing file")

if (os.path.exists(path + 'mapUserId-new')):
    mapUserId = pickle.load(open(path + 'mapUserId-new','rb'))
elif (os.path.exists(path + 'mapUserId-old')):
    mapUserId = pickle.load(open(path + 'mapUserId-old','rb'))
else:
    raise NameError("Missing file")

if (os.path.exists(path + 'mapPlaceId-new')):
    mapPlaceId = pickle.load(open(path + 'mapPlaceId-new','rb'))
elif (os.path.exists(path + 'mapPlaceId-old')):
    mapPlaceId = pickle.load(open(path + 'mapPlaceId-old','rb'))
else:
    raise NameError("Missing file")

mapPlaceId = pickle.load(open(path + 'mapPlaceId-new','rb'))

pred = np.array(pred_for_all_user[mapUserId[userId]])

max_id = np.where(pred == np.amax(pred))[0]

if (len(max_id) > 30):
    result = np.random.choice(min_id, 30, replace = False)
else:
    result = pred.argsort()[::-1][:30]

convertPlaceId = list(mapPlaceId.keys())

for i in range(len(result)):
    result[i] = convertPlaceId[result[i]]


res = ''
for i in range(len(result)-1):
    res += str(result[i]) + ','

res += str(result[len(result)-1])

print(res)
   
