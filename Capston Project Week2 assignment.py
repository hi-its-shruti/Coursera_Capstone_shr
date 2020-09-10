#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import cufflinks as cf
cf.go_offline()
import warnings
import datetime as dt


# In[5]:


df_ilc_holidays = pd.read_excel('/Learning/Idata.xlsx', sheet_name='Sheet1')
#df_ilc_holidays['Weekending Date'] = df_ilc_holidays['Weekending Date'].apply(lambda x: dt.datetime.strptime(x,'%m/%d/%Y').strftime('%Y-%m-%d'))
df_ilc_holidays.set_index('Weekending Date',inplace=True)
df_ilc_holidays


# In[7]:


p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))
pdq


# In[9]:


seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
seasonal_pdq


# In[10]:


warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.SARIMAX(endog=df_ilc_holidays.loc[:'2020-06-19']['Mean hours'],exog=df_ilc_holidays.loc[:'2020-06-19']['Holiday'],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        print('SARIMAX{}x{}12 - AIC {}'.format(param,param_seasonal,results.aic))


# In[9]:


# SARIMAX(1, 1, 1)x(0, 1, 1, 12)12 - AIC 355.7550553219099
mod = sm.tsa.SARIMAX(endog=df_ilc_holidays.loc[:'2020-06-19']['Mean hours'],exog=df_ilc_holidays.loc[:'2020-06-19']['Holiday'],order=(1,1,1),seasonal_order=(0,1,1,12),enforce_invertibility=False,enforce_stationarity=False)
results = mod.fit()
print(results.summary())


# In[10]:


results.plot_diagnostics(figsize=(15,12))
plt.show()


# In[11]:


# make ILC predictions for 2019 using already existing data
pred_2019 = results.get_prediction(start='2019-01-04',end='2019-12-27')
pred_ci = pred_2019.conf_int()


# In[12]:


df_ilc_2019 = pd.DataFrame()
df_ilc_2019['Actual'] = df_ilc_holidays.loc['2019-01-04':'2019-12-27']['Mean hours']
df_ilc_2019['Predictions'] = pred_2019.predicted_mean
df_ilc_2019['Deviations'] = np.abs(df_ilc_2019['Actual']-df_ilc_2019['Predictions'])
df_ilc_2019


# In[13]:


rmse_2019 = np.sqrt((df_ilc_2019['Deviations']**2).mean())
rmse_2019


# In[14]:


df_ilc_2019[['Actual','Predictions']].iplot(title='ILC for year 2019',xTitle='Week',yTitle='Efforts in hours')


# In[15]:


# make ILC predictions for 2020 using already existing data
pred_2020 = results.get_prediction(start='2020-01-03',end='2020-06-19')
pred_ci = pred_2020.conf_int()


# In[17]:


df_ilc_2020 = pd.DataFrame()
df_ilc_2020['Actual'] = df_ilc_holidays.loc['2020-01-03':'2020-06-19']['Mean hours']
df_ilc_2020['Predictions'] = pred_2020.predicted_mean
df_ilc_2020


# In[18]:


rmse_2020 = np.sqrt(((df_ilc_2020['Actual']-df_ilc_2020['Predictions'])**2).mean())
rmse_2020


# In[19]:


# make predictions for the entire year
forecast_2020 = results.get_forecast(steps=27,exog=df_ilc_holidays.loc['2020-06-26':'2020-12-25']['Holiday'])
forecast_2020.predicted_mean


# In[20]:


df_2020 = pd.DataFrame()
df_2020['Predictions'] = forecast_2020.predicted_mean
df_2020


# In[22]:


# concat the 2 dataframes
df_concat = pd.concat([df_ilc_2020,df_2020], sort=False)
df_concat


# In[23]:


# as we see some index as Dates while some others as DateTime, we will convert it into date
df_concat.index = pd.to_datetime(df_concat.index)
df_concat.index.name = 'Weekending Date'
df_concat


# In[25]:


df_concat.iplot(title='Actual vs Forecast ILC for year 2020',xTitle='Week',yTitle='Efforts in hours')


# In[27]:


df_concat.to_csv('Actual vs Forecast ILC 2020.csv')


# In[ ]:




