import sys
import errno
import os
import json
import logging
import random



from functools import wraps
from constant import REGISTRATION_KEY
from utils import make_response, make_error, get_cipher, gen_token, verify_token, run_command

from flask import Flask, Blueprint, request, jsonify
from werkzeug.contrib.fixers import ProxyFix

import pymongo
import PlanningTrip.pso_algorithm
from Recommender.Criteria_Based_RS import get_recommender_by_criteria 
from Recommender.MF_recommender import get_recommender_by_MF
from Recommender.Matrix_Factorization_5models_time_tlike import train_MF

from flask.json import JSONEncoder
from bson import ObjectId, json_util
from mongoengine.base import BaseDocument
from mongoengine.queryset.base import BaseQuerySet

class MongoEngineJSONEncoder(JSONEncoder):
    def default(self,obj):
        if isinstance(obj,BaseDocument):
            return json_util._json_convert(obj.to_mongo())
        elif isinstance(obj,BaseQuerySet):
            return json_util._json_convert(obj.as_pymongo())
        if isinstance(obj, ObjectId):
            return str(obj)
        return JSONEncoder.default(self, obj)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

# init app
app = Flask(__name__, static_folder=None)
app.json_encoder = MongoEngineJSONEncoder


# App modes
if app.debug is True:
    app.secret_key = 'A0Zr9YRFHY@%J R~XHH!jmN]LWX/,?RT'
else:
    secret_key_path = os.path.join(THIS_DIR, 'secret_key')
    try:
        app.secret_key = open(secret_key_path, 'rb').read()
    except IOError as exc:
        if errno.ENOENT == exc.errno:
            print('authenticator.py cannot find {}.'.format(secret_key_path))
            print('Create it with \npython -c '
                  "'import os; print(os.urandom(32))' > {}".format(secret_key_path))
            sys.exit(1)
        raise exc

# Authorization method
def authorization(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        headers = request.headers
        if not headers.get('Authorization'):
            return make_error("Unauthorized", 401)
        key = headers.get('Authorization')
        if key == REGISTRATION_KEY:
            return f(*args, **kwargs)
        else:
            return make_error('Invalid key', 403)
    return wrapper


def authenticate(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        headers = request.headers
        if headers and not headers.get('Authorization'):
            return make_error("Unauthorized", 401)
        token = verify_token(headers.get('Authorization'))
        return f(token.get('sub'), *args, **kwargs)
    return wrapper

# reading place info from database
local_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
guidy_db = local_mongo["guidy"]
place_info_collections = guidy_db["place_info"]

PlanningTrip.pso_algorithm.place_info = {}

for row in place_info_collections.find():
    PlanningTrip.pso_algorithm.place_info[str(row['placeId'])] = row
    # PlanningTrip.pso_algorithm.place_info.append(row)

if len(PlanningTrip.pso_algorithm.place_info) > 0:
    print("!!!!Connect Guidy database successfully!!!!")

# Routes
@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/route', methods=['POST'])
@authorization
def planning_tour():
    body = json.loads(request.data.decode("utf-8"))

    restaurant_list = body["restaurantList"]

    travel_list =  body["travelList"]

    planning = body["planning"]

    optimal_routes = PlanningTrip.pso_algorithm.pso_route_generate(planning, restaurant_list, travel_list )
    
    response_data = {
        "routes": optimal_routes
    }   
    return jsonify(response_data)
    
    # return jsonify(JSONEncoder().encode(response_data))

@app.route('/planning-trips', methods=['POST'])
@authorization
def planning_trips():
    print("POST /planning-trips")
    body = json.loads(request.data.decode("utf-8"))
    
    userId = body['userId']

    if "criteria" in body:
        restaurant_list = get_recommender_by_criteria(body["criteria"]).tolist()
    else:
        restaurant_list = get_recommender_by_MF(userId).tolist()

    travel_places = list(place_info_collections.find({"type": "VISITING"}, {"_id": 0, "placeId": 1}))

    travel_list = random.sample([row["placeId"] for row in travel_places], 100)

    planning = body["planning"]

    optimal_routes = PlanningTrip.pso_algorithm.pso_route_generate(planning, restaurant_list, travel_list )
    
    response_data = {
        "routes": optimal_routes
    }
    print('Done')
    return jsonify(response_data)

@app.route('/recommender-places/criteria', methods=['POST'])
@authorization
def get_criteria_recommender_places():
    print("POST /recommender-places/criteria")
    body = json.loads(request.data.decode("utf-8"))
    criteria = body['criteria']
    result = get_recommender_by_criteria(criteria)
    return make_response({"recommenderPlaces":result.tolist()})


@app.route('/recommender-places/MF-recommender', methods=['POST'])
@authorization
def get_recommender_MF():
    print("POST /recommender-places/MF-recommender")
    body = json.loads(request.data.decode("utf-8"))
    userId = body['userId']
    place_type = body['placeType']
    result = get_recommender_by_MF(userId, place_type)
    return make_response({"recommenderPlaces": result.tolist()})

    


@app.route('/recommender-places/train-model', methods=['POST'])
@authorization
def train_model_tlike():
    body = json.loads(request.data.decode("utf-8"))
    place_type = body['placeType']
    train_MF(place_type)
    return make_response({"message": "Training sucessfully"})


if __name__ == '__main__':
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    app.run(host='0.0.0.0', port=8086)