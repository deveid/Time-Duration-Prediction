# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 04:57:21 2018

@author: David
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy.stats import skew
from scipy import stats
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from xgboost.sklearn import XGBRegressor
import xgboost as xgb

sns.set_style('whitegrid')
train='file:///C:/Users/David/Documents/train.csv'
test='file:///C:/Users/David/Documents/test.csv'
train=pd.read_csv(train)
test=pd.read_csv(test)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])

#sns.barplot(data=train,'trip_duration')
train = train[train['trip_duration'] <= m + s]
train = train[train['trip_duration'] >= m - s]
#Limiting the coordnates to NY (city_long_border = (-74.03, -73.75),city_lat_border = (40.63, 40.85) )
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]    
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]
#Changing datetime from object to date 
train['pickup_datetime']=pd.to_datetime(train.pickup_datetime)
test['pickup_datetime']=pd.to_datetime(test.pickup_datetime)
train['dropoff_datetime']=pd.to_datetime(train.dropoff_datetime)
train['pickup_date']=train['pickup_datetime'].dt.date
train['pickup_time']=train['pickup_datetime'].dt.time
test['pickup_date']=test['pickup_datetime'].dt.date
test['pickup_time']=test['pickup_datetime'].dt.time
train['dropoff_date']=train['dropoff_datetime'].dt.date
train['droppoff_time']=train['dropoff_datetime'].dt.time

#dropping dropoff_datetime & pickup_datetime
train.drop(['pickup_datetime','dropoff_datetime'],axis=1)
plt.hist(train['trip_duration'],bins=1000)
plt.xlabel('Trip Duration')
plt.show()
#We can see a right skew , so we use the log transformation
train['log_tripduration']=(np.log(train['trip_duration'])+1)
#test['log_tripduration']=(np.log(test['trip_duration'])+1)

plt.plot(train.groupby('pickup_date')['trip_duration'].count())
plt.xlabel('Pickup Date')
plt.title('Trip Time')
plt.show()#we can see that the pickuptime btw late Jan& early feb is fast.What could be the reason 
plt.plot(train.groupby('dropoff_date')['trip_duration'].count())
plt.xlabel('Dropoff Date')
plt.title('Dropoff Time')
plt.show()#we can see that the pickuptime btw late Jan& early feb is fast.What could be the reason
sns.distplot(train['log_tripduration'],fit=norm)

#f,ax=plt.subplots(1,1,figsize=(15,8))
#Vendor per time
vendor_per_time=train.groupby('vendor_id')['trip_duration'].mean()
sns.barplot(vendor_per_time.index,vendor_per_time.values)
plt.title('Vendor Per Time')
plt.show()
#Store and fwd flag against time
store_per_time=train.groupby('store_and_fwd_flag')['trip_duration'].mean()
sns.barplot(store_per_time.index,store_per_time.values)
plt.title('Store/Flag Per time')
plt.show()

#Exploring further vendor id that store and forwARD AND ALSO NOT STORE 
Notstore_flag=train['store_and_fwd_flag']=='N'
store_flag=train['store_and_fwd_flag']=='Y'
Notstore_flag=train['vendor_id'][Notstore_flag]
sns.countplot(Notstore_flag)
plt.title('Vendors vehicle that had no connection to the server')
plt.show()
store_flag=train['vendor_id'][store_flag]
sns.countplot(store_flag)
plt.title('Vendors vehicle that had connection to the server')
plt.show()# We noticed only "Vendor 1" had good connection to the server
#Visualizing the map for our NY border region
NYcity_long_border,NYcity_lat_border=(-74.03, -73.75),(40.63, 40.85)
#f,ax=plt.subplots(ncols=2,sharex=True)
#sns.swarmplot(x=train['pickup_longitude'].values,y=train['pickup_latitude'].values,label='train',alpha=0.1,color='blue',ax=ax[0])
#sns.swarmplot(x=test['pickup_longitude'].values,y=test['pickup_latitude'].values,label='test',color='red',alpha=0.1,ax=ax[1])
#f.suptitle('Pickup Longitude and Latitude Train/Test')
#ax[0].xlim(NYcity_long_border)
#ax[0].ylim(NYcity_lat_border)
#ax[1].xlim(NYcity_long_border)
#ax[1].ylim(NYcity_lat_border)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.show()
#Extracting the date into day,hour,month
train['day']=train['pickup_datetime'].dt.day
test['day']=test['pickup_datetime'].dt.day
train['hour']=train['pickup_datetime'].dt.hour
test['hour']=test['pickup_datetime'].dt.hour
train['month']=train['pickup_datetime'].dt.month
test['month']=test['pickup_datetime'].dt.month
train['dayofweek']=train['pickup_datetime'].dt.dayofweek
test['dayofweek']=test['pickup_datetime'].dt.dayofweek

#Calculating the distance btw the lat and long 
def haversine(lat1,lat2,long1,long2):
    lat1,lat2,long1,long2=np.radians((lat1,lat2,long1,long2))
    lat=lat2-lat1
    long=long2-long1
    radius_of_earth=6371  #in km
    d=np.sin(lat*0.5)**2+np.cos(lat1)*np.cos(lat2)*np.sin(long*0.5)**2
    d=2*radius_of_earth*np.arcsin(np.sqrt(d))
    return d

train['haversine_distance']=haversine(train['pickup_latitude'].values,train['dropoff_latitude'].values,train['pickup_longitude'].values,train['dropoff_longitude'].values)
train['speed']=1000*train['haversine_distance']/train['trip_duration']
#Average speed during hours,days,months
f,ax=plt.subplots(ncols=3,sharey=True)
ax[0].plot(train.groupby('dayofweek').mean()['speed'],color='blue',alpha=0.7,lw=2)
ax[1].plot(train.groupby('hour').mean()['speed'],color='red',alpha=0.7)
ax[2].plot(train.groupby('month').mean()['speed'],color='green',alpha=0.7)
plt.suptitle('Average speed during hours,days,months in NY region')
ax[0].set_ylabel('Speed')
ax[0].set_xlabel('Day of the week')
ax[1].set_xlabel('Hour of the day')
ax[2].set_xlabel('Month of the Year')
plt.show()

vi_train=pd.get_dummies(train['vendor_id'],prefix='vi',prefix_sep='_')
vi_test=pd.get_dummies(test['vendor_id'],prefix='vi',prefix_sep='_')
pc_train=pd.get_dummies(train['passenger_count'],prefix='pc',prefix_sep='_')
pc_test=pd.get_dummies(test['passenger_count'],prefix='pc',prefix_sep='_')
sf_train=pd.get_dummies(train['store_and_fwd_flag'],prefix='sf',prefix_sep='_')
sf_test=pd.get_dummies(test['store_and_fwd_flag'],prefix='sf',prefix_sep='_')

train_day=pd.get_dummies(train['day'],prefix='d',prefix_sep='_')
test_day=pd.get_dummies(test['day'],prefix='d',prefix_sep='_')
train_dow=pd.get_dummies(train['dayofweek'],prefix='dow',prefix_sep='_')
test_dow=pd.get_dummies(test['dayofweek'],prefix='dow',prefix_sep='_')
train_h=pd.get_dummies(train['hour'],prefix='h',prefix_sep='_')
test_h=pd.get_dummies(test['hour'],prefix='h',prefix_sep='_')
train_m=pd.get_dummies(train['month'],prefix='m',prefix_sep='_')
test_m=pd.get_dummies(test['month'],prefix='m',prefix_sep='_')

df=pd.concat([vi_train,pc_train,sf_train,train_day,train_dow,train_h,train_m,train['log_tripduration']],axis=1)
df_test=pd.concat([vi_test,pc_test,sf_test,test_day,test_dow,test_h,test_m],axis=1)
df_test.drop('pc_9',inplace=True,axis=1)
#Splitting Train and test
Train, Test = train_test_split(df[0:1000000], test_size = 0.3)

X_train = Train.drop(['log_tripduration'], axis=1)
Y_train = Train["log_tripduration"]
X_test = Test.drop(['log_tripduration'], axis=1)
Y_test = Test["log_tripduration"]
#Creating matrices for Xgboost
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(df_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
#Tweaking xgboost metrics
xgb_pars = {'min_child_weight': 1, 'eta': 0.6, 'colsample_bytree': 0.8, 
            'max_depth': 8,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=7,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)
xgb.plot_importance(model, height=0.7)
pred = model.predict(dtest)
pred = np.exp(pred) - 1
test_id=test['id']
submission=pd.DataFrame({'Id':test_id,'trip_duration':pred})
#submission.to_csv('submis.csv')