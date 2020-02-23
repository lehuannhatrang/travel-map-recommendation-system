
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
from scipy import sparse 
import sys
import pickle

userId = list(map(int, sys.argv[1:]))[0]

path = '/home/yntn/Thesis/travel-map-recommendation-system/'

pred_for_all_user = pickle.load(open(path + 'MF_pred_for_all_user-new','rb'))
mapUserId = pickle.load(open(path + 'mapUserId-new','rb'))
mapPlaceId = pickle.load(open(path + 'mapPlaceId-new','rb'))

pred = np.array(pred_for_all_user[mapUserId[userId]])
result = pred.argsort()[::-1][:15]
convertPlaceId = list(mapPlaceId.keys())

for i in range(len(result)):
    result[i] = convertPlaceId[result[i]]

print(result)