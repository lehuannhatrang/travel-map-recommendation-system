import pandas as pd 
import numpy as np
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse 

class MF(object):
    """docstring for CF"""
    def __init__(self, n_users, n_items, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
            learning_rate = 0.5, max_iter = 100, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization parameter
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0

        # self.n_users = int(np.max(Y_data[:, 0])) + 1 
        # self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_users = int(n_users) + 1
        self.n_items = int(n_items) + 1

        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: # from saved data
            self.W = Winit
            
        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()

    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: # item base
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = (self.Y_raw_data[:, user_col] ).astype(np.int32)
        self.mu = np.zeros((n_objects,))
        for n in range(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = (self.Y_data_n[ids, item_col]).astype(np.int32)
            # and the corresponding ratings 
            ratings = self.Y_data_n[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_n[ids, 2] = ratings - self.mu[n]

    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2

    # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L 


    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where((self.Y_data_n[:,0]).astype(np.int32) == user_id)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)


    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where((self.Y_data_n[:,1]).astype(np.int32) == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)


    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))

    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                        self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
            
    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                # print ('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)


        # model = NMF(n_components=10, init='random', random_state=0)
        # self.W = model.fit_transform(X)
        # self.H = model.components_

    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 10]
        if pred < 1:
            return 1 
        if pred > 10: 
            return 10 
        return pred 


    def pred_for_user(self, user_id):
        """
        predict ratings one user give all unrated items
        """
        ids = np.where((self.Y_data_n[:, 0]).astype(np.int32) == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              

        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings= []

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))

        return predicted_ratings
        
    def pred_for_all_user(self):
        all_pre_rating = []

        for user_id in range (self.n_users):
            ids = np.where((self.Y_data_n[:, 0]).astype(np.int32) == user_id)[0]
            y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id] 
            all_pre_rating.append(y_pred.tolist())

        return all_pre_rating 

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE



tmp = pd.read_json('../data/user_rates_place-ver2.json')

# ratings_x = tmp[['User_Id','Place_Id','Rating','TimeStamp']]
ratings_x = tmp[['User_Id','Place_Id','Rating']]

userId = ratings_x.User_Id.unique()
userId.sort()
mapUserId  = {}
for i, value in enumerate(userId):
  mapUserId[value] = i

placeId = ratings_x.Place_Id.unique()
placeId.sort()
mapPlaceId  = {}
for i, value in enumerate(placeId):
  mapPlaceId[value] = i

tmp_based = ratings_x.values

for i in range(len(tmp_based[:,0])):
  tmp_based[i][0] = mapUserId[tmp_based [i][0]]
for i in range(len(tmp_based[:,1])):
  tmp_based[i][1] = mapPlaceId[tmp_based [i][1]]

tmp_based_copy = tmp_based


tmp_based_copy = tmp_based_copy[tmp_based_copy[:,0].argsort()] # sort theo  user_id
based_train = []
based_test = []
_usersId = (tmp_based_copy[:, 0]).astype(np.int32)

for i in range((max(tmp_based_copy[:,0])).astype(int)):  
  _ids = np.where(_usersId == i)[0].astype(np.int32)
  if (len(_ids) > 4):
    X_train, X_test= train_test_split(tmp_based_copy[_ids], test_size=.2, random_state=16)
    based_train += X_train.tolist()
    based_test += X_test.tolist()
    
based_train = np.array(based_train)
based_test = np.array(based_test)
# X_test = np.array(X_test)

rs = MF(int(max(tmp_based_copy[:,0])), int(max(tmp_based_copy[:,1])), based_train, K = 10, lam = .1, print_every = 10, learning_rate = 0.75, max_iter = 100, user_based = 1)
rs.fit()



##### store the predict matrix

pre = rs.pred_for_all_user()
path = '/home/yntn/Thesis/travel-map-recommendation-system/'
if (os.path.exists(path + 'MF_pred_for_all_user-new')):
    os.rename(path + 'MF_pred_for_all_user-new', path + 'MF_pred_for_all_user-old')
    pickle.dump(pre, open(path + 'MF_pred_for_all_user-new', 'wb'))    
    os.remove(path + 'MF_pred_for_all_user-old')
else:
    pickle.dump(pre, open(path + 'MF_pred_for_all_user-new', 'wb'))

if (os.path.exists(path + 'mapUserId-new')):
    os.rename(path + 'mapUserId-new', path + 'mapUserId-old')
    pickle.dump(mapUserId, open(path + 'mapUserId-new', 'wb')) 
    os.remove(path + 'mapUserId-old')   
else:
    pickle.dump(mapUserId, open(path + 'mapUserId-new', 'wb'))

if (os.path.exists(path + 'mapPlaceId-new')):
    os.rename(path + 'mapPlaceId-new', path + 'mapPlaceId-old')
    pickle.dump(mapPlaceId, open(path + 'mapPlaceId-new', 'wb')) 
    os.remove(path + 'mapPlaceId-old')  
else:
    pickle.dump(mapPlaceId, open(path + 'mapPlaceId-new', 'wb'))