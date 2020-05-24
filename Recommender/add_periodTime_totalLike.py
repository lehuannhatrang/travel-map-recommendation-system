import pandas as pd 
import numpy as np
import datetime

tmp = pd.read_json('../Crawl_Data/user_rates_place-ver2.json')

# ratings_x = tmp[['User_Id','Place_Id','Rating','TimeStamp']]
ratings_x = tmp[['User_Id','Place_Id', 'Rating', 'Rating_Space',	'Rating_Location',	'Rating_Quality',	'Rating_Service',	'Rating_Price']]
ratings_x_ts  = tmp[['TimeStamp']]


ts = datetime.datetime.now().timestamp()

time_period = ratings_x_ts.values
go = []
for i in range (len(time_period)):
  go.append(ts - (time_period[i][0] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

ratings_x.insert(2, 'Period_Time', go, True)