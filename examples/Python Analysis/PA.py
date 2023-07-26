#!/usr/bin/env python
# coding: utf-8

# 1.Consider the data from the Federal Reserve Economic Database (FRED) which is
# accessible using the link, https://fred.stlouisfed.org/series/IPG2211A2N concerning the Industrial Production Index for Electricity and Gas Utilities from January 2000 to Jan 2023. Perform the following tasks.
# 

# In[1]:


pip install jupyter


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv("IPG2211A2N (1).csv",index_col=0,parse_dates=True)


# In[4]:


df


# a. Resample the series and choose an essential graph to visualise the month wise
# five number summary.

# In[5]:


month = df.resample('M').mean()
summary = month.describe().loc[['min', '25%', '50%', '75%', 'max']]


# In[6]:


plt.figure(figsize=(20, 10))
summary.plot.line()
plt.xlabel('Monthly')
plt.ylabel('Industrial Production Index')
plt.title('Month-wise Five-Number Summary')
plt.show()


# In[7]:


summary


# In[8]:


df["Month"]=pd.DatetimeIndex(df.index).month


# In[9]:


fig,ax=plt.subplots(figsize=(11,8))
sns.barplot(data=df,x="Month",y="IPG2211A2N",ax=ax)
ax.set_title("IPG2211A2N production")
ax.set_ylabel("industrial production of electric and gas utilities")


# In[10]:


plt.style.use('fivethirtyeight')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 5


# In[11]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# b.Enumerate the components of time series using additive and multiplicative time series analysis

# In[12]:


rcParams["figure.figsize"]=10,12
result_additive=seasonal_decompose(df["IPG2211A2N"],model="additive")
result_additive.plot()


# In[13]:


rcParams["figure.figsize"]=10,12
result_additive=seasonal_decompose(df["IPG2211A2N"],model="multiplicative")
result_additive.plot()


# 2.Is the dataset used in Q.1 is stationary? If yes, justify. Otherwise, make the time series stationary. Also, using illustrative plots comment on why ARIMA model requires a time series to be stationary for forecasting.

# In[14]:


df['Year'] = pd.DatetimeIndex(df.index).year
df['Day Name'] = pd.DatetimeIndex(df.index).day_name()


# In[15]:


df["Year"].nunique()


# In[16]:


for year in np.unique(df.Year):
    X=df.loc[str(year)]["IPG2211A2N"].values
    result=adfuller(X)
    print("\n for the year:",year)
    print("ADF Statistic: %f"% result[0])
    print("p-value: %f" % result[1])
    for key,value in result[4].items():
        print("\t%5s: %8.3f" %(key,value))
X=df["IPG2211A2N"].values
result=adfuller(X)
print("\n for the consolidated 24 years")
print("ADF statistic %f" % result[0])
print("p-value %f" %result[1])
print("critical values")
for key,value in result[4].items():
    print("\t%5s %8.3f"%(key,value))


# OBSERVATION : From the final statement we can understand that this data is stationary as the p-value is more than 0.05

# In[17]:


ts=month["IPG2211A2N"]


# In[18]:


plt.figure(figsize=(12,6))
plt.plot(ts)
plt.title('Original Series')
acf0= plot_acf(ts)  #no differencing


# In[19]:


plt.plot(ts.diff());
plt.title("1st order Differencing")
acf1=plot_acf(ts.diff().dropna())


# In[20]:


plt.plot(ts.diff().diff());
plt.title("2nd order Differencing")
acf1=plot_acf(ts.diff().diff().dropna())


#  For instance, the widely used ARIMA (Auto-Regressive Integrated Moving Average) model for forecasting makes the assumption that the data is stationary. The model will not be able to faithfully reflect the underlying patterns in the data if it is non-stationary, and the findings will be erroneous. As an illustration, a non-stationary time series could give the impression that there is a high connection between two variables, but in reality, the correlation is only there because of a trend or a seasonal element in the data.
# 
# Stationary time series is when the mean and variance are constant over time. It is easier to predict when the series is stationary. Differencing is a method of transforming a non-stationary time series into a stationary one. This is an important step in preparing data to be used in an ARIMA model.

# 3. The management of an organisation would like to know the worldwide online visitors 
# browsing the pages and search engines of their website. The log entries of the users are 
# stored in a .csv file 'Log_Reg_dataset.csv'. The dataset has various features like 
# Country, Age, Repeat_Visitor, Search Engine, Web pages Viewed, and Status. Perform 
# descriptive statistics of each feature using PySpark. Assign probability value 0 and 1 
# for class 0 and class 1 respectively. Import the required machine learning packages in 
# PySpark and fit a logistic regression model to predict the Status of the test dataset. 
# Compute the confusion matrix and report your observation on the classification 
# metrics.

# In[21]:


import pyspark


# In[22]:


from pyspark.sql import SparkSession


# In[23]:


spark=SparkSession.builder.appName("Log_reg").getOrCreate()


# In[24]:


data=spark.read.csv("Log_Reg_dataset.csv",inferSchema=True,header=True)
data.show()


# In[25]:


data.printSchema()


# In[26]:


data.describe().show()


# In[27]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[28]:


data.columns


# In[29]:


data.groupBy("Country").count().show()


# In[30]:


data.groupBy("Status").count().show()


# In[31]:


from pyspark.ml.feature import StringIndexer
search_engine_indexer=StringIndexer(inputCol="Status",outputCol="Status No")
fit=search_engine_indexer.fit(data)
data=fit.transform(data)


# In[32]:


from pyspark.ml.feature import OneHotEncoder


# In[33]:


search_engine_encoder=OneHotEncoder(inputCol="Status No",outputCol="Status_vector")
search_engine_encoder.setDropLast(False)
fit=search_engine_encoder.fit(data)
data=fit.transform(data)


# In[34]:


data.show(3,truncate=False)


# In[35]:


country_indexer=StringIndexer(inputCol="Country",outputCol="Country no")


# In[36]:


fit=country_indexer.fit(data)
data=fit.transform(data)


# In[37]:


country_encoder=OneHotEncoder(inputCol="Country no",outputCol="Country_encoder")
ohe=country_encoder.fit(data)
data=ohe.transform(data)


# In[38]:


data.show(3,truncate=False)


# In[39]:


from pyspark.ml.feature import VectorAssembler


# In[40]:


data_assembler=VectorAssembler(
    inputCols=["Status_vector","Country_encoder","Age","Repeat_Visitor","Web_pages_viewed"],outputCol="features")
data=data_assembler.transform(data)


# In[41]:


data.select(["features","Status"])


# In[42]:


model_data=data.select(["features","Status"])


# In[43]:


from pyspark.ml.classification import LogisticRegression


# In[44]:


training_data,test_data=model_data.randomSplit([0.80,0.20])


# In[45]:


training_data.count()


# In[46]:


training_data.groupBy("Status").count().show()


# In[47]:


test_data.count()


# In[48]:


test_data.groupBy("Status").count().show()


# In[49]:


log_reg = LogisticRegression(labelCol='Status').fit(training_data)


# In[50]:


train_results = log_reg.evaluate(training_data).predictions 


# In[51]:


train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).select(['Status','prediction','probability']).show(10,False)


# In[52]:


correct_preds = train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).count()
correct_preds


# In[53]:


results = log_reg.evaluate(test_data).predictions


# In[54]:


results.select(['Status','prediction']).show(20,False)


# In[55]:


results.printSchema()


# In[56]:


results[(results['Status']==1) & (results.prediction ==1)].count()


# In[107]:


tp = results[(results['Status']==1)& (results.prediction==1)].count()
tn = results[(results['Status']==0)& (results.prediction==0)].count()
fp = results[(results['Status']==0)& (results.prediction==1)].count()
fn = results[(results['Status']==1)& (results.prediction==0)].count()
print(tp,fp,tn,fn)


# OBSERVATION : By calculating we can get the values for the given confusion matrix

# In[108]:


precision=tp/(tp+fp)
print("precision",precision)


# In[109]:


recall=tp/(tp+fn)
print("recall",recall)


# In[110]:


accuracy=(tp+tn)/(tp+tn+fp+fn)
print("accuracy",accuracy)


# In[112]:


trainsummary = log_reg.summary
roc = trainsummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel("True Positive Rate")
plt.title('ROC Curve')
plt.show()


# In[62]:


print('Training set area Under the curve (AUC): '+str(trainsummary.areaUnderROC))


# OBSERVATION : From the final statement we can observe that 1.0 is the common value for precision,accuracy and recall

# 4.Given the dataset “Superstore.xlsx”, forecast the sales of the category “Office 
# Supplies” by performing the tasks given below.
# a. Explore the dataset to report Stationarity using Rolling plots and Statistical 
# Tests.
# b. Determine the order of differencing, d.
# c. Determine the order p for autoregressive process Αℛ(𝑝).
# d. Determine the order q for moving averages 𝑀𝐴(𝑞).
# e. Fit ARIMA (p, d, q) using Q.4.b, 4.c, and 4.d and report the results with 
# accuracy metrics.

# In[70]:


store=pd.read_excel("Superstore.xls")
store.info()


# In[71]:


store["Category"].unique()


# In[72]:


store.Category.value_counts()


# In[73]:


store.Region.value_counts()


# In[74]:


ofs=store.loc[store["Category"]=="Office Supplies"]
ofs


# In[75]:


ofs.columns


# a.check if more than one sales data is there for a date

# In[113]:


os=ofs.groupby('Order Date')["Sales"].size()
os


# In[78]:


os[os>1].sort_values(ascending=False)


# In[114]:


os=ofs.set_index("Order Date")
os


# In[81]:


plt.figure(figsize=(16,6))
os.plot()


# In[115]:


os["Year"]=pd.DatetimeIndex(os.index).year
os


# In[117]:


os["Month"]=pd.DatetimeIndex(os.index).month
os["Day"]=pd.DatetimeIndex(os.index).day


# In[118]:


os["Sales"].plot()


# In[119]:


import seaborn as sns
fig,ax=plt.subplots(figsize=(11,8))
sns.boxplot(data=os,x="Month",y="Sales",ax=ax)
ax.set_title("Sales")
ax.set_label("Daily consolidated sales")


# In[120]:


cols_plot=['Sales']
os.loc['2014'][cols_plot].plot(linewidth=0.5)
os.loc['2015'][cols_plot].plot(linewidth=0.5)
os.loc['2016'][cols_plot].plot(linewidth=0.5)
os.loc['2017'][cols_plot].plot(linewidth=0.5)


# In[121]:


#monthly sales
data_columns=["Sales"]
os_week=os[data_columns].resample("W").sum()
os_week.head()


# In[122]:


cols_plot=['Sales']
os_week.loc['2014'][cols_plot].plot(linewidth=0.5)
os_week.loc['2015'][cols_plot].plot(linewidth=0.5)
os_week.loc['2016'][cols_plot].plot(linewidth=0.5)
os_week.loc['2017'][cols_plot].plot(linewidth=0.5)


# In[123]:


data_colums=['Sales']
os_month=os[data_colums].resample('M').sum()
os_month.head()


# In[91]:


cols_plot=['Sales']
os_month.loc['2014'][cols_plot].plot(linewidth=0.5)
os_month.loc['2015'][cols_plot].plot(linewidth=0.5)
os_month.loc['2016'][cols_plot].plot(linewidth=0.5)
os_month.loc['2017'][cols_plot].plot(linewidth=0.5)


# OBSERVATION : We can see that there is a slight uptrend in the month of june to december

# In[124]:


from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# In[125]:


for year in np.unique(os.Year):
    #detect_trend(data_grouped.loc[str(year)]['Sales'].values)

    X = os.loc[str(year)]['Sales'].values

    result = adfuller(X)
   
    print('\nFor the year : ', year)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%5s: %8.3f' % (key, value))

X = os['Sales'].values

result = adfuller(X)

print('\nFor the CONSOLIDATED 4 years')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%5s: %8.3f' % (key, value))


# In[126]:


plt.rcParams.update({"figure.figsize":(9,3),"figure.dpi":120})
plt.plot(os_month.diff())
plt.title("1st differencing")
pacf=plot_pacf(os_month.diff().dropna())


# OBSERVATION : From these two plot , we don't see any spikes in lower order therefore we can try with 0 or 1 later

# In[127]:


#Order of the MA terms(q)
#fig,axes=plt.subplots(1,2,share=True)
plt.plot(os_month.diff())
plt.title("1st differencing")
#axes[1].set(ylim=(0,1.2))
acf=plot_acf(os_month.diff().dropna())


# OBSERVATION : The given graph suggest that the data is not stationary as the p-value is less then 0.05

# In[128]:


from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


# In[129]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


plt.figure(figsize=(12,6))
plt.plot(os_month)
plt.title("original_series")
acf0=plot_acf(os_month)#no differencing


# In[131]:


plt.plot(os_month)
plt.title("1st order differencing")
acf1=plot_acf(os_month.dropna())


# In[132]:


plt.plot(os_month.diff().diff())
plt.title("2nd order differencing")
acf2=plot_acf(os_month.diff().diff().dropna())


# OBSERVATION : We can't find any differences between the graph

# C.Find the order of the AR term

# In[133]:


plt.rcParams.update({"figure.figsize":(9,3),"figure.dpi":120})
plt.plot(os_month)
plt.title("1st differencing")
pacf=plot_pacf(os_month.diff().dropna())


# D.Find the order of the MA term(q):

# In[134]:


plt.plot(os_month)
plt.title("1st Differencing")
acf=plot_acf(os_month.diff().dropna())


# OBSERVATION : Also with the lag 1 the autocorrelation plot is going negative so we give q as 0
# 
# Defintion : S is seasonal arima :X stands for exogenous forecast

# E. Fit ARIMA (p, d, q) using Q.4.b, 4.c, and 4.d and report the results with
# accuracy metrics.

# In[135]:


mod_default=sm.tsa.statespace.SARIMAX(os_month,order=(1,1,1),enfore_stationarity=False,enforce_invertibility=False)
results_default=mod_default.fit()


# In[136]:


print(results_default.summary())


# OBSERVATION : From the above observation the p-value for the ar.L1 is greater than 0.05 ,the ma.L1 model is less than 0.05 so ma.L1 model gets fitted conveniently 

# In[ ]:




