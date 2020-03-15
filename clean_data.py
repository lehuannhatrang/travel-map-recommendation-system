import pandas as pd
import numpy as np

path = '/home/yntn/Thesis/'

a = pd.read_csv(open('Data/restaurant-foody-user-reviews-filter-ver4.csv','r'))

b = pd.read_json(open('Data/user_rates_place-ver2.json','rb'))

x  = pd.concat([a,b], ignore_index=True)

x.drop_duplicates(subset=['User_Id', 'Place_Id'],keep="first",inplace=True)

#delete row which rating_avg = 0 and list rating is all NaN
count = []
for i in range (x.shape[0]):
    rating = x.Rating.iloc[i]*5    
    avg = (x.Rating_Location.iloc[i] + x.Rating_Price.iloc[i] + x.Rating_Quality.iloc[i]  + x.Rating_Service.iloc[i] + x.Rating_Space.iloc[i])
    if (rating != avg):
        if ((pd.isnull(x.iloc[i]['Rating_Space']) & pd.isnull(x.iloc[i]["Rating_Location"] ) & pd.isnull(x.iloc[i]["Rating_Quality"] ) & pd.isnull(x.iloc[i]["Rating_Service"]) & pd.isnull(x.iloc[i]["Rating_Price"]))):
            if rating == 0:
                count.append(x.index[i])

x = x.drop(count)

count1 = []
       
for i in range (x.shape[0]):
    rating = x.Rating.iloc[i]*5    
    avg = (x.Rating_Location.iloc[i] + x.Rating_Price.iloc[i] + x.Rating_Quality.iloc[i]  + x.Rating_Service.iloc[i] + x.Rating_Space.iloc[i])
    if (rating > avg):
        count1.append(x.index[i])
        

for i in range (len(count1)):
    x.at[count1[i], 'Rating_Space'] = x.Rating[count1[i]]    
    x.at[count1[i], 'Rating_Location'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Quality'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Service'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Price'] = x.Rating[count1[i]]

count1 = []

for i in range (x.shape[0]):
    rating = x.Rating.iloc[i]*5    
    avg = (x.Rating_Location.iloc[i] + x.Rating_Price.iloc[i] + x.Rating_Quality.iloc[i]  + x.Rating_Service.iloc[i] + x.Rating_Space.iloc[i])
    if (rating != avg):
        if ((pd.isnull(x.iloc[i]['Rating_Space']) & pd.isnull(x.iloc[i]["Rating_Location"] ) & pd.isnull(x.iloc[i]["Rating_Quality"] ) & pd.isnull(x.iloc[i]["Rating_Service"]) & pd.isnull(x.iloc[i]["Rating_Price"]))):
            count1.append(x.index[i])

for i in range (len(count1)):
    x.at[count1[i], 'Rating_Space'] = x.Rating[count1[i]]    
    x.at[count1[i], 'Rating_Location'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Quality'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Service'] = x.Rating[count1[i]]
    x.at[count1[i], 'Rating_Price'] = x.Rating[count1[i]]

count2 = []
for i in range (x.shape[0]):
    rating = x.Rating.iloc[i]*5    
    avg = (x.Rating_Location.iloc[i] + x.Rating_Price.iloc[i] + x.Rating_Quality.iloc[i]  + x.Rating_Service.iloc[i] + x.Rating_Space.iloc[i])
    if (rating < avg):
        count2.append(x.index[i])
  
for i in range(len(count2)):
    x.at[count2[i], 'Rating'] = (x.Rating_Space[count2[i]] + x.Rating_Location[count2[i]] + x.Rating_Quality[count2[i]] + x.Rating_Service[count2[i]] + x.Rating_Price[count2[i]])/5

x.to_json(r'Data/user_rates_place-100MB.json')

x["weight"] = x["TotalLike"]+1

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

newdf = (x.groupby("Place_Id")["weight"]).sum().reset_index()

newdf = newdf.drop(["weight"], axis =1)

Lst = ["Rating", "Rating_Space", "Rating_Location", "Rating_Quality", "Rating_Service", "Rating_Price"]
for i in range(6):
    newdf.insert(i+1, Lst[i],(x.groupby("Place_Id").apply(wavg, Lst[i], "weight")).values)

newdf.to_json(r'Data/user_rates_place-100MB-final.json')