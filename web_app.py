# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 20:28:33 2022

@author: dhiraj
"""

import pandas as pd
import category_encoders as ce
import lightgbm as lgb
import pickle
import streamlit as st
from datetime import datetime
import numpy as np


st.title("Welcome to the Revenue Prediction System")
st.subheader('Please Upload all the past Data of an User')
uploaded_file = st.file_uploader("",key = "1")

if not uploaded_file:
  st.stop()
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    

def predict(df):
    
    allowed_columns = ['channelGrouping', 'date', 'fullVisitorId', 'visitNumber', 'visitStartTime', 'device.browser',
                       'device.operatingSystem','device.isMobile', 'device.deviceCategory', 'geoNetwork.continent',
                       'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region', 'geoNetwork.metro', 
                       'geoNetwork.city', 'geoNetwork.networkDomain', 'totals.hits', 'totals.pageviews', 
                       'totals.bounces', 'totals.newVisits', 'trafficSource.campaign',
                       'trafficSource.source', 'trafficSource.medium', 'trafficSource.keyword',
                       'trafficSource.isTrueDirect', 'trafficSource.referralPath']
     
    
    for column in df.columns:
        if column not in allowed_columns:
            df = df.drop(column, axis=1)
    st.header("The Uploaded Data")
    st.write(df)
            
    #print(df.columns)
    df['date'] = pd.to_datetime(df["date"], infer_datetime_format=True, format="%Y%m%d")
    df['weekday'] = df.date.dt.weekday
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year' ] = df.date.dt.year
    
    def process_time(posix_time):
        return datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%d %H:%M:%S')
    def process_hour(time):
        return str(time)[-8:-6]

    df['visitStartTime'] = df['visitStartTime'].apply(process_time)
    df['visitHour'] = df['visitStartTime'].apply(process_hour)
    categorical_columns=['channelGrouping','device.browser','device.operatingSystem','device.isMobile','device.deviceCategory','geoNetwork.continent','geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region','geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.networkDomain','totals.bounces','totals.newVisits','trafficSource.campaign','trafficSource.source','trafficSource.medium', 'trafficSource.keyword','trafficSource.isTrueDirect','trafficSource.referralPath','weekday','day','month','year','visitHour']

    file = open("encoder.obj",'rb')
    encoder = pickle.load(file)
    file.close()

    df[categorical_columns] = encoder.transform( df[categorical_columns] )

    df['totals.pageviews'].fillna(2.0,inplace=True)
    #print(pd.DataFrame(df))
    

    grouped_df = df.groupby('fullVisitorId').agg({ 'totals.pageviews':[('total_pageviews_max',lambda x : x.dropna().max()),
                                                                   ('total_pageviews_min',lambda x : x.dropna().min()), 
                                                                   ('total_pageviews_mean',lambda x : x.dropna().mean()),
                                                                   ('total_pageviews_mode',lambda x : x.value_counts().index[0])],
                                           
                                     'channelGrouping': [('channelGrouping_max',lambda x : x.dropna().max()),
                                                         ('channelGrouping_min',lambda x : x.dropna().min()),
                                                         ('channelGrouping_mode',lambda x : x.value_counts().index[0])],
                                           
                                     'visitNumber': [('visitNumber_max',lambda x : x.dropna().max()),
                                                     ('visitNumber_mean',lambda x : x.dropna().mean()),
                                                     ('visitNumber_min',lambda x : x.dropna().min())],
                                           
                                     'device.browser':[('device_browser_max',lambda x : x.dropna().max()),
                                                       ('device_browser_min',lambda x : x.dropna().min()),
                                                       ('device_browser_mode',lambda x : x.value_counts().index[0])],
                                           
                                    'device.operatingSystem':[('device_operatingSystem_max',lambda x : x.dropna().max()),
                                                              ('device_operatingSystem_min',lambda x : x.dropna().min()),
                                                              ('device_operatingSystem_mode',lambda x : x.value_counts().index[0])],
                                   
                                     'device.isMobile':[('device_isMobile_max',lambda x : x.dropna().max()),
                                                        ('device_isMobile_min',lambda x : x.dropna().min())],
                                           
                                   'device.deviceCategory':[('device_deviceCategory_max',lambda x : x.dropna().max()),
                                                            ('device_deviceCategory_min',lambda x : x.dropna().min()),
                                                            ('device_deviceCategory_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'geoNetwork.continent':[('geoNetwork_continent_max',lambda x : x.dropna().max()),
                                                           ('geoNetwork_continent_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.subContinent':[('geoNetwork_subContinent_max',lambda x : x.dropna().max()),
                                                              ('geoNetwork_subContinent_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.country':[('geoNetwork_country_max',lambda x : x.dropna().max()),
                                                         ('geoNetwork_country_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.region':[('geoNetwork_region_max',lambda x : x.dropna().max()),
                                                        ('geoNetwork_region_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.metro':[('geoNetwork_metro_max',lambda x : x.dropna().max()),
                                                       ('geoNetwork_metro_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.city':[('geoNetwork_city_max',lambda x : x.dropna().max()),
                                                      ('geoNetwork_city_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.networkDomain':[('geoNetwork_networkDomain_max',lambda x : x.dropna().max()),
                                                               ('geoNetwork_networkDomain_min',lambda x : x.dropna().min()),
                                                               ('geoNetwork_networkDomain_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'totals.hits':[('totals_hits_max',lambda x : x.dropna().max()),
                                                  ('totals_hits_min',lambda x : x.dropna().min()),
                                                  ('totals_hits_mean',lambda x : x.dropna().mean())],
                                           
                                   'totals.bounces':[('totals_bounces_max',lambda x : x.dropna().max()),
                                                     ('totals_bounces_min',lambda x : x.dropna().min()),
                                                     ('totals_bounces_mean',lambda x : x.dropna().mean())],
                                           
                                   'totals.newVisits':[('totals_newVisits_max',lambda x : x.dropna().max()),
                                                       ('totals_newVisits_min',lambda x : x.dropna().min())],
                                           
                                   'trafficSource.campaign':[('trafficSource_campaign_max',lambda x : x.dropna().max()),
                                                             ('trafficSource_campaign_min',lambda x : x.dropna().min()),
                                                             ('trafficSource_campaign_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.source':[('trafficSource_source_max',lambda x : x.dropna().max()),
                                                           ('trafficSource_source_min',lambda x : x.dropna().min()),
                                                           ('trafficSource_source_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.medium':[('trafficSource_medium_max',lambda x : x.dropna().max()),
                                                           ('trafficSource_medium_min',lambda x : x.dropna().min()),
                                                           ('trafficSource_medium_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.keyword':[('trafficSource_keyword_max',lambda x : x.dropna().max()),
                                                            ('trafficSource_keyword_min',lambda x : x.dropna().min())],
                                           
                                   'trafficSource.isTrueDirect':[('trafficSource_isTrueDirect_max',lambda x : x.dropna().max()),
                                                                 ('trafficSource_isTrueDirect_min',lambda x : x.dropna().min()),
                                                                 ('trafficSource_isTrueDirect_mean',lambda x : x.dropna().mean())],
                                           
                                   'trafficSource.referralPath':[('trafficSource_referralPath_max',lambda x : x.dropna().max()),
                                                                 ('trafficSource_referralPath_min',lambda x : x.dropna().min()),
                                                                 ('trafficSource_referralPath_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'weekday':[('weekday_max',lambda x : x.dropna().max()),
                                              ('weekday_min',lambda x : x.dropna().min()),
                                              ('weekday_mean',lambda x : x.dropna().mean()),
                                              ('weekday_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'day':[('day_max',lambda x : x.dropna().max()),
                                          ('day_min',lambda x : x.dropna().min()),
                                          ('day_mean',lambda x : x.dropna().mean()),
                                          ('day_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'month':[('month_max',lambda x : x.dropna().max()),
                                            ('month_min',lambda x : x.dropna().min()),
                                            ('month_mean',lambda x : x.dropna().mean()),
                                            ('month_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'year':[('year_max',lambda x : x.dropna().max()),
                                           ('year_min',lambda x : x.dropna().min()),
                                           ('year_mode',lambda x : x.value_counts().index[0])], 
                                           
                                   'visitHour':[('visitHour_max',lambda x : x.dropna().max()),
                                                ('visitHour_mean',lambda x : x.dropna().mean()),
                                                ('visitHour_min',lambda x : x.dropna().min()),
                                                ('visitHour_mode',lambda x : x.value_counts().index[0])]     
                                    })
    grouped_df.columns = grouped_df.columns.droplevel()
    grouped_df = grouped_df.reset_index()

    loaded_model = pickle.load(open("saved_lgbm.sav", 'rb'))
    predicted_value = loaded_model.predict(grouped_df.drop(['fullVisitorId'], axis=1).values.reshape(1, -1))
    
    return round(predicted_value[0],3)


prediction = str(predict(dataframe))
st.subheader("The Estimated Revenue that the user can generate = "+prediction)
st.text('Note : This is Log of Sum of all the Revenue that the user will generate during the next year')

######################################################################################################

def model_performance(df):
    
    allowed_columns = ['channelGrouping', 'date', 'fullVisitorId', 'visitNumber', 'visitStartTime', 'device.browser',
                       'device.operatingSystem','device.isMobile', 'device.deviceCategory', 'geoNetwork.continent',
                       'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region', 'geoNetwork.metro', 
                       'geoNetwork.city', 'geoNetwork.networkDomain', 'totals.hits', 'totals.pageviews', 
                       'totals.bounces', 'totals.newVisits', 'trafficSource.campaign',
                       'trafficSource.source', 'trafficSource.medium', 'trafficSource.keyword',
                       'trafficSource.isTrueDirect', 'trafficSource.referralPath','totals.transactionRevenue']
     
    
    for column in df.columns:
        if column not in allowed_columns:
            df = df.drop(column, axis=1)
            
    #print(df.columns)
    df['date'] = pd.to_datetime(df["date"], infer_datetime_format=True, format="%Y%m%d")
    df['weekday'] = df.date.dt.weekday
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year' ] = df.date.dt.year
    
    def process_time(posix_time):
        return datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%d %H:%M:%S')
    def process_hour(time):
        return str(time)[-8:-6]

    df['visitStartTime'] = df['visitStartTime'].apply(process_time)
    df['visitHour'] = df['visitStartTime'].apply(process_hour)
    categorical_columns=['channelGrouping','device.browser','device.operatingSystem','device.isMobile','device.deviceCategory','geoNetwork.continent','geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region','geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.networkDomain','totals.bounces','totals.newVisits','trafficSource.campaign','trafficSource.source','trafficSource.medium', 'trafficSource.keyword','trafficSource.isTrueDirect','trafficSource.referralPath','weekday','day','month','year','visitHour']

    file = open("encoder.obj",'rb')
    encoder = pickle.load(file)
    file.close()

    df[categorical_columns] = encoder.transform( df[categorical_columns] )

    df['totals.pageviews'].fillna(2.0,inplace=True)
    #print(pd.DataFrame(df))
    

    grouped_df = df.groupby('fullVisitorId').agg({ 'totals.pageviews':[('total_pageviews_max',lambda x : x.dropna().max()),
                                                                   ('total_pageviews_min',lambda x : x.dropna().min()), 
                                                                   ('total_pageviews_mean',lambda x : x.dropna().mean()),
                                                                   ('total_pageviews_mode',lambda x : x.value_counts().index[0])],
                                           
                                     'channelGrouping': [('channelGrouping_max',lambda x : x.dropna().max()),
                                                         ('channelGrouping_min',lambda x : x.dropna().min()),
                                                         ('channelGrouping_mode',lambda x : x.value_counts().index[0])],
                                           
                                     'visitNumber': [('visitNumber_max',lambda x : x.dropna().max()),
                                                     ('visitNumber_mean',lambda x : x.dropna().mean()),
                                                     ('visitNumber_min',lambda x : x.dropna().min())],
                                           
                                     'device.browser':[('device_browser_max',lambda x : x.dropna().max()),
                                                       ('device_browser_min',lambda x : x.dropna().min()),
                                                       ('device_browser_mode',lambda x : x.value_counts().index[0])],
                                           
                                    'device.operatingSystem':[('device_operatingSystem_max',lambda x : x.dropna().max()),
                                                              ('device_operatingSystem_min',lambda x : x.dropna().min()),
                                                              ('device_operatingSystem_mode',lambda x : x.value_counts().index[0])],
                                   
                                     'device.isMobile':[('device_isMobile_max',lambda x : x.dropna().max()),
                                                        ('device_isMobile_min',lambda x : x.dropna().min())],
                                           
                                   'device.deviceCategory':[('device_deviceCategory_max',lambda x : x.dropna().max()),
                                                            ('device_deviceCategory_min',lambda x : x.dropna().min()),
                                                            ('device_deviceCategory_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'geoNetwork.continent':[('geoNetwork_continent_max',lambda x : x.dropna().max()),
                                                           ('geoNetwork_continent_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.subContinent':[('geoNetwork_subContinent_max',lambda x : x.dropna().max()),
                                                              ('geoNetwork_subContinent_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.country':[('geoNetwork_country_max',lambda x : x.dropna().max()),
                                                         ('geoNetwork_country_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.region':[('geoNetwork_region_max',lambda x : x.dropna().max()),
                                                        ('geoNetwork_region_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.metro':[('geoNetwork_metro_max',lambda x : x.dropna().max()),
                                                       ('geoNetwork_metro_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.city':[('geoNetwork_city_max',lambda x : x.dropna().max()),
                                                      ('geoNetwork_city_min',lambda x : x.dropna().min())],
                                           
                                   'geoNetwork.networkDomain':[('geoNetwork_networkDomain_max',lambda x : x.dropna().max()),
                                                               ('geoNetwork_networkDomain_min',lambda x : x.dropna().min()),
                                                               ('geoNetwork_networkDomain_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'totals.hits':[('totals_hits_max',lambda x : x.dropna().max()),
                                                  ('totals_hits_min',lambda x : x.dropna().min()),
                                                  ('totals_hits_mean',lambda x : x.dropna().mean())],
                                           
                                   'totals.bounces':[('totals_bounces_max',lambda x : x.dropna().max()),
                                                     ('totals_bounces_min',lambda x : x.dropna().min()),
                                                     ('totals_bounces_mean',lambda x : x.dropna().mean())],
                                           
                                   'totals.newVisits':[('totals_newVisits_max',lambda x : x.dropna().max()),
                                                       ('totals_newVisits_min',lambda x : x.dropna().min())],
                                           
                                   'trafficSource.campaign':[('trafficSource_campaign_max',lambda x : x.dropna().max()),
                                                             ('trafficSource_campaign_min',lambda x : x.dropna().min()),
                                                             ('trafficSource_campaign_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.source':[('trafficSource_source_max',lambda x : x.dropna().max()),
                                                           ('trafficSource_source_min',lambda x : x.dropna().min()),
                                                           ('trafficSource_source_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.medium':[('trafficSource_medium_max',lambda x : x.dropna().max()),
                                                           ('trafficSource_medium_min',lambda x : x.dropna().min()),
                                                           ('trafficSource_medium_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'trafficSource.keyword':[('trafficSource_keyword_max',lambda x : x.dropna().max()),
                                                            ('trafficSource_keyword_min',lambda x : x.dropna().min())],
                                           
                                   'trafficSource.isTrueDirect':[('trafficSource_isTrueDirect_max',lambda x : x.dropna().max()),
                                                                 ('trafficSource_isTrueDirect_min',lambda x : x.dropna().min()),
                                                                 ('trafficSource_isTrueDirect_mean',lambda x : x.dropna().mean())],
                                           
                                   'trafficSource.referralPath':[('trafficSource_referralPath_max',lambda x : x.dropna().max()),
                                                                 ('trafficSource_referralPath_min',lambda x : x.dropna().min()),
                                                                 ('trafficSource_referralPath_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'weekday':[('weekday_max',lambda x : x.dropna().max()),
                                              ('weekday_min',lambda x : x.dropna().min()),
                                              ('weekday_mean',lambda x : x.dropna().mean()),
                                              ('weekday_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'day':[('day_max',lambda x : x.dropna().max()),
                                          ('day_min',lambda x : x.dropna().min()),
                                          ('day_mean',lambda x : x.dropna().mean()),
                                          ('day_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'month':[('month_max',lambda x : x.dropna().max()),
                                            ('month_min',lambda x : x.dropna().min()),
                                            ('month_mean',lambda x : x.dropna().mean()),
                                            ('month_mode',lambda x : x.value_counts().index[0])],
                                           
                                   'year':[('year_max',lambda x : x.dropna().max()),
                                           ('year_min',lambda x : x.dropna().min()),
                                           ('year_mode',lambda x : x.value_counts().index[0])], 
                                           
                                   'visitHour':[('visitHour_max',lambda x : x.dropna().max()),
                                                ('visitHour_mean',lambda x : x.dropna().mean()),
                                                ('visitHour_min',lambda x : x.dropna().min()),
                                                ('visitHour_mode',lambda x : x.value_counts().index[0])],
                                                  
                                   'totals.transactionRevenue':[('revenue_sum',lambda x : x.dropna().sum())]

                                    })
    grouped_df.columns = grouped_df.columns.droplevel()
    grouped_df = grouped_df.reset_index()
    
    grouped_df['log_Revenue'] = np.log1p(grouped_df['revenue_sum'])
    grouped_df.drop('revenue_sum', axis=1, inplace = True)


    loaded_model = pickle.load(open("saved_lgbm.sav", 'rb'))
    
    
    return np.sqrt(np.sum(np.square(loaded_model.predict(grouped_df.drop(['log_Revenue','fullVisitorId'], axis=1))-grouped_df['log_Revenue']))/len(grouped_df.drop(['log_Revenue','fullVisitorId'], axis=1)))
    

    
st.subheader('Please Upload all the Data to get RMSE score')
uploaded_file_ = st.file_uploader("",key = "2")

if not uploaded_file_:
  st.stop()
if uploaded_file_ is not None:
    dataframe_ = pd.read_csv(uploaded_file_)

rmse = str(model_performance(dataframe_))
st.subheader("The Root Mean Squared Error on Data = "+rmse)
