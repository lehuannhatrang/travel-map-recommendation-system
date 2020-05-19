import sys
from flask import Flask, Blueprint, request
from argparse import ArgumentParser
from werkzeug.contrib.fixers import ProxyFix
import pymongo
import pso_algorithm
import json


planning_trip = Blueprint('Planning_Trip', __name__)

# reading place info from database
local_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
guidy_db = local_mongo["guidy"]
place_info_collections = guidy_db["place_info"]

pso_algorithm.place_info =  []
for row in place_info_collections.find():
    pso_algorithm.place_info.append(row)

@planning_trip.route('/')
def hello_world():
    return "Hello, World!"

@planning_trip.route('/route', methods=['POST'])
def planning_tour():
    body = json.loads(request.data)

    restaurant_list = body["restaurantList"]

    travel_list =  body["travelList"]

    planning = body["planning"]

    optimal_routes = pso_algorithm.pso_route_generate(planning, restaurant_list, travel_list )
    
    response_data = {
        "route": optimal_routes
    }
    return str(response_data)


if __name__ == '__main__':

    # arg parser for the standard anaconda-project options
    parser = ArgumentParser(prog="Planning_trip",
                            description="Plaaning Trip module for Guidy")
    parser.add_argument('--anaconda-project-host', action='append', default=[],
                        help='Hostname to allow in requests')
    parser.add_argument('--anaconda-project-port', action='store', default=8086, type=int,
                        help='Port to listen on')
    parser.add_argument('--anaconda-project-iframe-hosts',
                        action='append',
                        help='Space-separated hosts which can embed us in an iframe per our Content-Security-Policy')
    parser.add_argument('--anaconda-project-no-browser', action='store_true',
                        default=False,
                        help='Disable opening in a browser')
    parser.add_argument('--anaconda-project-use-xheaders',
                        action='store_true',
                        default=False,
                        help='Trust X-headers from reverse proxy')
    parser.add_argument('--anaconda-project-url-prefix', action='store', default='',
                        help='Prefix in front of urls')
    parser.add_argument('--anaconda-project-address',
                        action='store',
                        default='0.0.0.0',
                        help='IP address the application should listen on.')

    args = parser.parse_args()

    app = Flask(__name__)
    app.register_blueprint(planning_trip, url_prefix = args.anaconda_project_url_prefix)

    app.config['PREFERRED_URL_SCHEME'] = 'https'

    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host=args.anaconda_project_address, port=args.anaconda_project_port)