#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as rf_sk
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pyspark
from pyspark.ml.regression import RandomForestRegressor as rf_sp
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

sns.set(rc={'figure.figsize': (10, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')


# In[2]:


path = 'C:\\Users\\data'
full_path_list = [ path + '/' + f for                  f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]


# In[3]:


con = sql.connect("fitbit.db")
cur = con.cursor()

def get_table_name(full_path_list, i):
    return full_path_list[i].split("/")[-1].split('.')[0]

pbar = tqdm(total=len(full_path_list), desc='[Loading Data...]')
for i in range(0,len(full_path_list)):
    pd.read_csv(full_path_list[i]).to_sql(get_table_name(full_path_list, i), con, if_exists='append', index=False)
    pbar.update(1)
pbar.close()


# In[4]:


# simple sql query test
df = pd.read_sql(f'SELECT * FROM {get_table_name(full_path_list, 0)}', con)
df.head()


# In[5]:


# list all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print(f'Total of {len(tables)} tables in database.')


# In[10]:


# check for sample data and data size
fitbit_df = pd.read_sql(f'SELECT * FROM fitbit_df', con)

print(len(fitbit_df))

fitbit_df.head()


# In[13]:


#1) perform exploratory analysis
# Average Calories, Steps and Distance by Id and by day of the week
query = """
SELECT
	ROUND(AVG(Calories),2) AS avg_calories,
	ROUND(AVG(TotalSteps),2) AS avg_steps,
	ROUND(AVG(TotalMinutesAsleep),2) AS avg_minutesAsleep
FROM fitbit_df
GROUP BY TotalTimeInBed;
"""

activity_dist = pd.read_sql(query, con)
activity_dist.head()


# In[16]:


# join fitbit data and sleep data
join_query = """
SELECT
	A.SedentaryMinutes,
	S.TotalMinutesAsleep
FROM
	fitbit_df A
INNER JOIN sleepDay_merged S
ON
    A.TotalTimeInBed=S.TotalTimeInBed;
"""
activity_sleep_df = pd.read_sql(join_query, con)

activity_sleep_df.head()


# # a - d

# In[18]:


fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
fig.suptitle('calories burned per minute')

sns.regplot(data = fitbit_df, x = 'SedentaryMinutes', y = 'Calories', ax=axes[0])

sns.regplot(data = fitbit_df, x = 'TotalMinutesAsleep', y = 'Calories', ax=axes[1])

sns.regplot(data = fitbit_df, x = 'TotalSteps', y = 'Calories', ax=axes[2])

sns.regplot(data = fitbit_df, x = 'TotalTimeInBed', y = 'Calories', ax=axes[3])


# In[19]:


#correlation measures
column_to_move = fitbit_df.pop("Calories")
fitbit_df.insert(4, "Calories", column_to_move)
print(fitbit_df.columns)

# Define a function to plot the scatterplots of the relationships between 
# all independent variables and the dependent variable
def plot_relationships(df, num_cols):
    variables = df.columns
    
    # assume that the dependent variable is in the last column
    dep_var = variables[-1]
    ind_var = variables[:-1]
    figs = len(dep_var)
    num_cols = num_cols
    num_rows = round(figs / num_cols) + 1
    fig = 1
    plt.figure(figsize=(20,30))
    # Loop through all independent variables and create the scatter plot
    for i in ind_var:
        pltfignums = [str(num_rows), str(num_cols), str(fig)]
        pltfig = int(''.join(pltfignums))
        plt.subplot(pltfig)
        plt.scatter(df[i], df[dep_var])
        plt.xlabel(str(i))
        plt.ylabel(str(dep_var))
        fig +=1

plot_relationships(fitbit_df,4)


# In[23]:


# Plot the correlations as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(fitbit_df.corr(), annot=True, cmap='cividis', fmt='.2g')


# In[25]:


X_train_temp, X_test, y_train_temp, y_test = train_test_split(fitbit_df.iloc[:,:-1], 
                                                              fitbit_df['Calories'], 
                                                              test_size=0.2, 
                                                              random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, 
                                                      test_size=0.25, random_state=42)


# In[26]:



scaler = MinMaxScaler()

scaler.fit_transform(X_train)

scaler.transform(X_valid)
scaler.transform(X_test)


# In[27]:


def scoring(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mae = mean_absolute_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Mean Absolute Error: {:0.4f}.'.format(mae))
    print('Mean Squared Error: {:0.4f}.'.format(mse))
    print('R^2 Score = {:0.4f}.'.format(r2))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    

baseline_y = [y_train.median()] * len(y_valid)


base_predictions = baseline_y
base_mae = mean_absolute_error(y_valid, base_predictions)
base_mse = mean_squared_error(y_valid, base_predictions)
base_r2 = r2_score(y_valid, base_predictions)
base_errors = abs(base_predictions - y_valid)
base_mape = 100 * np.mean(base_errors / y_valid)
base_accuracy = 100 - base_mape
print('Model Performance')
print('Mean Absolute Error: {:0.4f}.'.format(base_mae))
print('Mean Squared Error: {:0.4f}.'.format(base_mse))
print('R^2 Score = {:0.4f}.'.format(base_r2))
print('Accuracy = {:0.2f}%.'.format(base_accuracy))


# In[28]:


# feature selection with lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

plt.figure(figsize=(10, 6))
plt.plot(range(len(X_train.columns)), lasso_coef)
plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=60)
plt.axhline(0.0, linestyle='--', color='r')
plt.ylabel('Coefficients')


# In[29]:


rf_regressor = rf_sk(random_state=42)
rf = rf_regressor.fit(X_train, y_train)

scoring(rf, X_valid, y_valid)


# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


dt_classifier=DecisionTreeClassifier()


# In[33]:


model=dt_classifier.fit(X_train,y_train)


# In[34]:


scoring(model,X_valid,y_valid)


# In[36]:


fn=['SedentaryMinutes', 'TotalSteps', 'TotalMinutesAsleep',
       'TotalTimeInBed']
cn=['Calories']
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)


# In[37]:


using PySpark and report your observations.
fitbit_df.to_csv('fitbit_df.csv', index=False)

CSV_PATH = "./fitbit_df.csv"
APP_NAME = "Random Forest"
SPARK_URL = "local[*]"
RANDOM_SEED = 13579
TRAINING_DATA_RATIO = 0.7
RF_NUM_TREES = 3
RF_MAX_DEPTH = 4
RF_NUM_BINS = 32

spark = SparkSession.builder     .appName(APP_NAME)     .master(SPARK_URL)     .getOrCreate()

df = spark.read     .options(header = "true", inferschema = "true")     .csv(CSV_PATH)

print("Total number of rows: %d" % df.count())

df.printSchema()

df.show()


# In[39]:


# create features
featureCols = ['SedentaryMinutes', 'TotalSteps', 'TotalMinutesAsleep',
       'TotalTimeInBed']
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
df = assembler.transform(df)
spark_df = df.select(['features', 'Calories'])

spark_df.show()
spark_df.printSchema()


# In[40]:


train, test = spark_df.randomSplit([0.8, 0.2], seed = 42)
print("Number of training set rows: %d" % train.count())
print("Number of test set rows: %d" % test.count())


# In[41]:


rf = rf_sp(featuresCol="features", labelCol='Calories')

model = rf.fit(train)

predictions = model.transform(test)

predictions.select("prediction", "Calories", "features").show(5)

preds = predictions.select('prediction').toPandas()['prediction']
test_labels = predictions.select('Calories').toPandas()['Calories']

errors = abs(preds - test_labels)
mape = 100 * np.mean(errors / test_labels)
accuracy = round(100 - mape, 4)
print(f'Accuracy = {accuracy} %')


# In[38]:


regressor = LinearRegression()
mlr = regressor.fit(X_train, y_train)
scoring(mlr, X_valid, y_valid)


# #A data scientist has to build an Artificial Neural Network (ANN) model consisting of
# two hidden layers with activation function Rectifier Unit and sigmoid function at the
# output layer for the dataset shown in Table.2. Design an ensemble workflow and Code
# the steps involved in building an ANN using tensorflow with batch size 34 and 100
# epochs. Also, predict if the customer with information shown in Table.2 will churn the
# telecom service:

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt


# In[76]:


pip install tensorflow


# In[80]:


get_ipython().run_line_magic('cd', 'D:\\predictive_data_analysis\\data')


# In[81]:


df=pd.read_csv("Telecom Customer Management.csv")


# In[82]:


df.head()


# In[96]:


df.columns


# In[103]:


df.dtypes


# In[101]:


from sklearn.preprocessing import LabelEncoder


# In[123]:


le=LabelEncoder()
df["gender"]=le.fit_transform(df["gender"])
df["Partner"]=le.fit_transform(df["Partner"])
df["Dependents"]=le.fit_transform(df["Dependents"])
df["PhoneService"]=le.fit_transform(df["PhoneService"])
df["MultipleLines"]=le.fit_transform(df["MultipleLines"])
df["InternetService"]=le.fit_transform(df["InternetService"])
df["OnlineSecurity"]=le.fit_transform(df["OnlineSecurity"])
df["OnlineBackup"]=le.fit_transform(df["OnlineBackup"])
df["DeviceProtection"]=le.fit_transform(df["DeviceProtection"])
df["TechSupport"]=le.fit_transform(df["TechSupport"])
df["StreamingTV"]=le.fit_transform(df["StreamingTV"])
df["StreamingMovies"]=le.fit_transform(df["StreamingMovies"])
df["Contract"]=le.fit_transform(df["Contract"])
df["PaperlessBilling"]=le.fit_transform(df["PaperlessBilling"])
df["PaymentMethod"]=le.fit_transform(df["PaymentMethod"])
df["TotalCharges"]=le.fit_transform(df["TotalCharges"])
df["Churn"]=le.fit_transform(df["Churn"])


# In[124]:


x=df.iloc[:,2:-1].values
y=df.iloc[:,-1].values


# In[125]:


x


# In[126]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
print(x)


# In[127]:


#split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[128]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[129]:


x_train.shape


# In[ ]:





# In[130]:


#initialize the ANN
ann=tf.keras.models.Sequential()
#adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6,input_shape=(19,),activation="relu"))
#adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


# In[135]:


#compiling the ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[136]:


#training the ANN on the training set
ann.fit(x_train,y_train,batch_size=34,epochs=100)


# In[143]:


df["gender"].nunique()
df["Partner"].nunique()
df["Dependents"].nunique()
df["PhoneService"].nunique()
df["MultipleLines"].nunique()
df["InternetService"].nunique()
df["OnlineSecurity"].nunique()
df["OnlineBackup"].nunique()
df["DeviceProtection"].nunique()
df["TechSupport"].nunique()
df["StreamingTV"].nunique()
df["StreamingMovies"].nunique()
df["Contract"].nunique()
df["PaperlessBilling"].nunique()
df["PaymentMethod"].nunique()
df["TotalCharges"].nunique()


# In[151]:


df.head()


# In[147]:


df["gender"].unique()


# In[152]:


df["MultipleLines"].unique()


# In[156]:


print(ann.predict(sc.transform([[1,1,0,1,0,2,0,0,1,0,0,1,0,0,0,1,2,59.85,39.85]]))>0.5)#standard scaler #0.5 is the threshold value for sigmoid function


# hence the predicted value for given input is false that means the customer will not churn

# In[157]:


#prediction and evaluation
y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[158]:


#making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# # 2. Consider a marketing plan in which the problem is to measure the impact of various ways of advertising on sales such as (1) YouTube, (2) Facebook, and (3) Newspaper.Hence, formulate a linear regression model using R programming for the dataset available in “devtools package” which can be installed using“devtools::install_github(“kassambra/datarium)” to solve the following predictive analytics tasks.

# Q.NO : 1-5

# In[ ]:


install.packages("readr")
library(readr)
install.packages("dplyr")
library(dplyr)
install.packages("Hmisc")
library(Hmisc)
install.packages("ggplot2")
library(ggplot2)
install.packages("datarium")
library(datarium)
install.packages("caret")
install.packages("magrittr")
library(magrittr)
library(caret)
data("marketing",package="datarium")
marketing_plan<-marketing
cor_yt_sales<-marketing_plan[,c("youtube","sales")]
cor(cor_yt_sales)

#OUTPUT:
youtube 1.0000000 0.7822244
sales   0.7822244 1.0000000


# In[ ]:


#2)
marketing_plan %>% ggplot(aes(x = youtube, y = sales)) +  geom_point() +
  labs(x = "Spending on YouTube ads",y = "Sales", title = "Graph 1: Relationship between YouTube ads and sales") +  stat_smooth(se = FALSE) +   theme(panel.background = element_rect(fill = "white", colour = "grey50"))
#there exist a positive relationship between youtube and sales
marketing_plan %>% ggplot(aes(x = facebook, y = sales)) +  geom_point() +
  labs(x = "Spending on Facebook",y = "Sales", title = "Graph 2: Relationship between Facebook and sales") +  stat_smooth(se = FALSE) +   theme(panel.background = element_rect(fill = "white", colour = "grey50"))
#there exist a positive relationship between facebook and sales
marketing_plan %>% ggplot(aes(x = newspaper, y = sales)) +  geom_point() +
  labs(x = "Spending on Newspaper",y = "Sales", title = "Graph 3: Relationship between Newspaper and sales") +  stat_smooth(se = FALSE) +   theme(panel.background = element_rect(fill = "white", colour = "grey50"))
#there exist a negative relationship between newspaper and sales


# In[ ]:


#3
Training and splitting 
set.seed(1)
train_indices<-createDataPartition(y=marketing[["sales"]],
                                   p=0.8,
                                   list=FALSE)
train_listings<-marketing[train_indices,]
test_listings<-marketing[-train_indices,]
model_0<-lm(sales~youtube+facebook+newspaper,data=train_listings)
summary(model_0)
model_1<-lm(sales~youtube+facebook,data=train_listings)
summary(model_1)


# In[ ]:


#4
model_2<-lm(sales~ facebook+I(facebook^2)+youtube+I(youtube^2),data=train_listings)
summary(model_2)
model_3<-lm(sales~facebook+poly(youtube,5),data=train_listings)
summary(model_3)


# In[ ]:


#5
model_4<- lm(sales~ facebook+poly(youtube,3)+facebook*youtube,data=train_listings)
summary(model_4)
#The combined effect of facebook and youtube will increases the sales
marketing_data <- marketing %>%select(youtube, facebook, newspaper, sales)
model <- lm(sales ~ youtube + facebook + newspaper, data = marketing_data)
new_data <- data.frame(youtube = 1000, facebook = 2000, newspaper = 1500)
predicted_sales <- predict(model, newdata = new_data)
summary(predicted_sales)
predicted_sales
marketing%>%select(sales)
ggplot(marketing_data, aes(x = youtube, y = sales)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(x = "Investment in YouTube", y = "Unit Sales") +
  theme_minimal()

ggplot(marketing_data, aes(x = facebook, y = sales)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "Investment in Facebook", y = "Unit Sales") +
  theme_minimal()

ggplot(marketing_data, aes(x = newspaper, y = sales)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "green") +
  labs(x = "Investment in Newspaper", y = "Unit Sales") +
  theme_minimal()

