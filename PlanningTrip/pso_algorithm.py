from __future__ import division
import random
import math

R = 6371000

A = 1

B = 1

C = 1

NUMBER_OF_BEST_ROUTE = 3

INERTIA_WEIGHT = 0.5

COGNATIVE_CONSTANT = 1

SOCIAL_CONSTANT = 2

PLACE_LIST = []

#------ FITNESS FUNCTION
def fitness_function(user_preference=0, distance=0):
    result = A*user_preference - B*distance/80000
    if result > -1 and result < 1:
        return result
    elif result > 1: 
        return 1
    elif result < -1:
        return -1 
# calculate distance between 2 coordinated
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_route_distance(route, planning, restaurant_list, travel_list):
    global place_info
    index = 0
    place_longitude = []
    place_latitude = []
    distances = []
    for place_index in route:
        if planning[index]["id"] == 0:
            place = next((x for x in place_info if str(x['placeId']) == restaurant_list[place_index]), None)
        else:
            place = next((x for x in place_info if str(x['placeId']) == travel_list[place_index]), None)

        current_place_longitude = float(place["longitude"])
        current_place_latitude = float(place["latitude"])
        # call api to estimate
        # use google clouds ~ 1000 reqs = 5$
        if index != 0:
            distance = calculate_distance((place_latitude[-1], place_longitude[-1]), (current_place_latitude, current_place_longitude))
            distances.append(distance)
        place_longitude.append(current_place_longitude)
        place_latitude.append(current_place_latitude)
        index += 1

    return sum(distances)

def checkTime(action_begin_time, action_end_time, placeId):
    global place_info
    place = next((x for x in place_info if str(x['placeId']) == placeId), None)

    action_begin_time_number = int(''.join(action_begin_time.split(':')))
    action_end_time_number = int(''.join(action_end_time.split(':')))
    place_begin_time_number = int(''.join(place["beginTime"].split('|')[0].split(':')))
    place_end_time_number = int(''.join(place["endTime"].split('|')[0].split(':')))

    if action_end_time_number - place_begin_time_number > 0 and place_end_time_number - action_end_time_number > 0 :
        return True
    else:
        return False

def random_initalize_position(planning, restaurant_list, travel_list):
    initalize_result = []
    for action_type in planning:
        while True:
            if action_type["id"] == 0:
                random_point = random.randint(0, len(restaurant_list) -1)
                place_id = restaurant_list[random_point]
            else:
                random_point = random.randint(0, len(travel_list)-1)
                place_id = travel_list[random_point]

            if random_point not in initalize_result and checkTime(action_type["beginTime"], action_type["endTime"], place_id):
                initalize_result.append(random_point)
                break

    return initalize_result


class Particle:
    def __init__(self, planning, restaurant_list, travel_list):
        self.position = random_initalize_position(planning, restaurant_list, travel_list)      # particle position
        self.velocity = []      # particle velocity
        self.pos_best_local = []    # local best position
        self.fitness_value_best = -99  # local fitness value
        self.fitness_value = -99  # local fitness value
        self.distance = 0
        self.user_preference = 0
        
        for i in range(0, len(self.position)):
            self.velocity.append(random.uniform(-1, 1))
            self.distance = get_route_distance(self.position, planning, restaurant_list, travel_list)
    
    # evalute current fitness
    def evaluate(self, fitness_func):
        self.fitness_value = fitness_func(user_preference=self.user_preference, distance=self.distance)
        # check and update local best
        if self.fitness_value > self.fitness_value_best:
            self.fitness_value_best = self.fitness_value
            self.pos_best_local = self.position

    # update particle velocity
    def update_velocity(self, pos_best_global):
        w = INERTIA_WEIGHT
        c1 = COGNATIVE_CONSTANT
        c2 = SOCIAL_CONSTANT

        for i in range(0, len(self.position)):
            r1 = random.random()
            r2 = random.random()
            # social velocity
            vel_cognitive = c1 * r1 * (self.pos_best_local[i] - self.position[i])
            # social velocity
            vel_social = c2 * r2 * (pos_best_global[i] - self.position[i])
            # update velocity
            self.velocity[i] = w * self.velocity[i] + vel_cognitive + vel_social

    # update position of particle
    def update_position(self, planning, restaurant_list, travel_list):
        for i in range(0,len(self.position)):
            new_position = int(self.position[i] + self.velocity[i])
            # bonus = -1 if new_position > self.pos_best_local[i] else 1
            while True:
                if planning[i]['id'] == 0:
                    restaurant_bounds = [0, len(restaurant_list) -1]
                    if new_position > restaurant_bounds[1]:
                        new_position = restaurant_bounds[1]
                        # bonus = 1
                    if new_position < restaurant_bounds[0]:
                        new_position = restaurant_bounds[0]
                        # bonus = -1
                    place_id = restaurant_list[new_position]
                else:
                    travel_bounds = [0, len(travel_list) - 1]
                    if new_position > travel_bounds[1]:
                        new_position = travel_bounds[1]
                        # bonus = 1
                    if new_position < travel_bounds[0]:
                        new_position = travel_bounds[0]
                        # bonus = -1
                    place_id = travel_list[new_position]

                if new_position not in self.position[:i] and checkTime(planning[i]["beginTime"], planning[i]["endTime"], place_id):
                    self.position[i] = new_position
                    break
                else: 
                    new_position += random.randint(-1,1)


class PSO():
    def __init__(self, fitness_func, planning, num_particles, maxiter, restaurant_list, travel_list):
        global num_dimensions
        self.best_routes = []
        self.best_route = []
        num_dimensions = len(planning)

        fitness_value_best_global = -99      # best fitness value of group particles
        pos_best_global = [0] * len(planning)                # best position of group particles
        routes = []
        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(planning, restaurant_list, travel_list))

        # begin loop
        index = 0
        while index < maxiter:
            for j in range(0, num_particles):
                # evaluate all particles fitness value
                swarm[j].evaluate(fitness_func)


                routes.append({
                    "route": swarm[j].position,
                    "fitness_value": swarm[j].fitness_value,
                    "distance": swarm[j].distance,
                    "user_perference": swarm[j].user_preference,
                })

                # update best global position and fitness value
                if swarm[j].fitness_value > fitness_value_best_global:
                    pos_best_global = swarm[j].position
                    # self.best_route = swarm[j].position
                    fitness_value_best_global = float(swarm[j].fitness_value)
            
            # update velocity and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_global)
                swarm[j].update_position(planning, restaurant_list, travel_list)

            is_converging = all(particle.position == swarm[0].position for particle in swarm)
            
            if is_converging:
                break
            index += 1

        best_routes = sorted(routes, key=lambda k: -k['fitness_value'])
        routes_value = []
        for route in best_routes:
            if route["route"] in routes_value:
                continue
            self.best_routes.append(route)
            routes_value.append(route["route"])


        print('Result: --------------')
        print(self.best_routes[:3])


NUMBER_PARTICLES = 10
MAX_ITER = 100

def pso_route_generate(planning, restaurant_list, travel_list):
    # global place_info
    pso = PSO(fitness_function, planning, NUMBER_PARTICLES, MAX_ITER, restaurant_list, travel_list )
    result = []
    index = 0
    for route in pso.best_routes[:NUMBER_OF_BEST_ROUTE]:
        place_ids = []
        for place in route['route']:
            if planning[index] == 0:
                place_ids.append(restaurant_list[place])
                # result.append(restaurant_list[place])
            else:
                place_ids.append(travel_list[place])
        route['route'] = place_ids
        result.append(route)
    return result

if __name__ == "__PSO__":
    main()