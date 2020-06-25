import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import *
import pymongo

import sys

local_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
guidy_db = local_mongo["guidy"]
place_rating_collections = guidy_db["place_rating"]


restaurant_criteria_data = []
for row in place_rating_collections.find({"type": "RESTAURANT"}):
    data = [row['placeId'], row['avgRating'], row['spaceRating'], row['locationRating'], row['qualityRating'], row['serviceRating'], row['priceRating']]
    restaurant_criteria_data.append(data)


visiting_criteria_data = []
for row in place_rating_collections.find({"type": "VISITING"}):
    data = [row['placeId'], row['avgRating'], row['spaceRating'], row['locationRating'], row['qualityRating'], row['serviceRating'], row['priceRating']]
    visiting_criteria_data.append(data)


def get_recommender_by_criteria(criteria, type="RESTAURANT"):
    if type == "RESTAURANT":
        dt = np.array(restaurant_criteria_data)
    elif type == "VISITING":
        dt = np.array(visiting_criteria_data)
    go = dt[:,-5:]
    vec = np.array([criteria]*len(go))
    sim = paired_distances(vec,go)

    min_id = np.where(sim == np.amin(sim))[0]

    if (len(min_id) > 100):
        result = np.random.choice(min_id, 100, replace = False)
    else:
        result = np.argsort(sim)[:100]

    return (dt[result,0]).astype(int)
   


