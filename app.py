import sys
import errno
import os
import json
import logging

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

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

# init app
app = Flask(__name__, static_folder=None)

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

PlanningTrip.pso_algorithm.place_info =  []
for row in place_info_collections.find():
    PlanningTrip.pso_algorithm.place_info.append(row)

if len(PlanningTrip.pso_algorithm.place_info) > 0:
    print("!!!!Connect Guidy database successfully!!!!")

# Routes
@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/route', methods=['POST'])
@authorization
def planning_tour():
    body = json.loads(request.data)

    restaurant_list = body["restaurantList"]

    travel_list =  body["travelList"]

    planning = body["planning"]

    optimal_routes = PlanningTrip.pso_algorithm.pso_route_generate(planning, restaurant_list, travel_list )
    
    response_data = {
        "planning": planning,
        "routes": optimal_routes
    }
    return jsonify(response_data)

@app.route('/recommender-places/criteria', methods=['POST'])
@authorization
def get_criteria_recommender_places():
    body = json.loads(request.data)
    criteria = body['criteria']
    result = get_recommender_by_criteria(criteria)
    return make_response({"recommenderPlaces":result.tolist()})


@app.route('/recommender-places/train-model', methods=['POST'])
@authorization
def train_model_tlike():
    body = json.loads(request.data)
    place_type = body['placeType']
    train_MF(place_type)
    return make_response({"message": "Training sucessfully"})


if __name__ == '__main__':
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    app.run(host='0.0.0.0', port=8086)