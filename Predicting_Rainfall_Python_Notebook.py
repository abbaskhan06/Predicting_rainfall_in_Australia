#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Implementing a predictive model on Rain Dataset to predict whether or not it will rain tomorrow in Australia

#  Dataset contains about 10 years of daily weather observations of different locations in Australia.

#  Problem Statement: Design a predictive model with the use of machine learning algorithms 
#                       to forecast whether or not it will rain tomorrow in Australia.

#  Data Source: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package:

#   Dataset Description:
#    Number of columns: 23
#    Number of rows: 145460
#    Number of Independent Columns: 22
#    Number of Dependent Column: 1


# In[2]:


# Contents:
#  Data Preprocessing
#  Finding categorical and Numerical features in Dataset
#  Cardinality check for categorical features
#  Handling Missing values
#  Outlier detection and treatment
#  Exploratory Data Analysis
#  Encoding categorical features
#  Correlation
#  Feature Importance
#  Splitting Data into Training and Testing sets
#  Feature Scaling
#  Model Building and Evaluation
#  Results and Conclusion
#  Save Model and Scaling object with Pickle


# In[3]:


#  Importing Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading the Dataset

dataset = 'weatherAUS.csv'
rain = pd.read_csv(dataset)

# Checking the Column Names 

print (rain.head())


# In[4]:


# Checking the Dimension of the Dataset:
#  As we can see from the results that the dataset has 23 variables and 145460 records

print(rain.shape)


# In[5]:


# Data Preprocessing: 


# Real-world data is often messy, incomplete, unstructured, inconsistent, redundant, sprinkled with wacky values. So, without deploying any Data Preprocessing techniques, it is almost impossible to gain insights from raw data.
# What exactly is Data Preprocessing?

#  Data preprocessing is a process of converting raw data to a suitable format to extract insights. 
#  It is the first and foremost step in the Data Science life cycle. 
#  Data Preprocessing makes sure that data is clean, organize and read-to-feed to the Machine Learning model.

# Consise Summary of the Dataset

print(rain.info())


# In[6]:


# We can see that except for the Date and The Location columns - every dataset has a missing value 
# We would now be generating "DESCRIPTIVE STATISTICS" of the dataset using the describe () function in pandas 
# The data values that are 'object' will be ommitted through the exclude = object arguement within describe() function

rain.describe(exclude=object).transpose()


# In[7]:


rain.describe(include=object).transpose()


# In[8]:


# Finding Categorical and Numerical Features in the Dataset

# Categorical Features in the Dataset

categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']

print("Number of Categorical Features: {}".format(len(categorical_features))) # Number of Categorical Features

print("Categorical Features:", categorical_features) # Names of Categorical Features


# In[9]:


# Numerical Feature in the Dataset

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']

print("Number of Numerical Features: {}".format(len(numerical_features)))

print("Numerical Features: ",numerical_features)


# In[10]:


# Cardinality check for Categorical features:

# The accuracy, performance of a classifier not only depends on the model that we use, 
# but also depends on how we preprocess data, and what kind of data you’re feeding to the classifier to learn.

# Many Machine learning algorithms like Linear Regression, Logistic Regression, k-nearest neighbors, etc. 
# can handle only numerical data, so encoding categorical data to numeric becomes a necessary step.  
# But before jumping into encoding, check the cardinality of each categorical feature.


# Cardinality: The number of unique values in each categorical feature is known as cardinality.


# A feature with a high number of distinct/ unique values is a high cardinality feature. 

# A categorical feature with hundreds of zip codes is the best example of a high cardinality feature.
#    This high cardinality feature poses many serious problems: 
#    Like it will increase the number of dimensions of data when that feature is encoded, which is not good for the model.
#    There are many ways to handle high cardinality, one would be feature engineering
#    The other is simply dropping that feature if it doesn’t add any value to the model.
#    Let’s find the cardinality for Categorical features:


# In[11]:


for each_feature in categorical_features:
   unique_values = len(rain[each_feature].unique())
   print("Cardinality(no. of unique values) of {} are: {}".format(each_feature, unique_values))


# In[12]:


# Date column has high cardinality which poses several problems to the model in terms of efficiency 
#   and also dimensions of data increase when encoded to numerical data.
# Feature Engineering of Date Column to decrease high Cardinality

rain['Date'] = pd.to_datetime(rain['Date'])
rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day

# Drop Date Column

rain.drop('Date', axis = 1, inplace = True)
rain.head()


# In[13]:


# Handling Missing Values in Categorical Features 

categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']

rain[categorical_features].isnull().sum()


# In[14]:


# Imputing the missing values in Categorical Variables Using the most frequent Value (MODE)

categorical_features_with_null = [feature for feature in categorical_features if rain[feature].isnull().sum()]

for each_feature in categorical_features_with_null:
    mode_val = rain[each_feature].mode()[0]
    rain[each_feature].fillna(mode_val,inplace=True)


# In[15]:


rain.Location.value_counts()


# In[16]:


# Handling Missing Values in Numerical Features

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']

rain[numerical_features].isnull().sum()


# In[17]:


# Missing Value in the Numerical Features can be Imputed as "Mean" or "Median".
# However, for this the outliers have to be addressed. 

# OUTLIER TREATMENT WITHIN NUMERICAL FEATURES

features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

# Using the Inter Quartile Range Rule to Address Outliers


for feature in features_with_outliers:
    q1 = rain[feature].quantile(0.25)
    q3 = rain[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    rain.loc[rain[feature]>upper_limit,feature] = upper_limit    


# In[18]:


# The Numerical Values are free from Outliers - Imputing the misisng vaues with numerical Mean

numerical_features_with_null = [feature for feature in numerical_features if rain[feature].isnull().sum()]
for feature in numerical_features_with_null:
    mean_value = rain[feature].mean()
    rain[feature].fillna(mean_value,inplace=True)


# In[19]:


# Its now time to do some Exploratory Data Analysis

rain['RainTomorrow'].value_counts().plot(kind='bar')


# In[20]:


# Looks like the Target variable is imbalanced. It has more ‘No’ values. If data is imbalanced, 
# then it might decrease the performance of the model. As this data is released by the meteorological 
# department of Australia, it doesn’t make any sense when we try to balance the target variable, 
# because the truthfulness of data might decrease. So, let me keep it as it is.

# Bi-Variate Analysis:
# Sunshine versus Rainfall 

sns.lineplot(data = rain, x='Sunshine', y='Rainfall', color = 'green')


# In[21]:


# Sunshine verus Rainfall

sns.lineplot(data = rain, x='Sunshine', y='Evaporation', color = 'blue')


# In[22]:


# Encoding of Categorical Variables -- Feature Encoding. 

# Most Machine Learning Algorithms like Logistic Regression, Support Vector Machines, 
# K Nearest Neighbours, etc. can’t handle categorical data. Hence, these categorical data need 
# to converted to numerical data for modeling, which is called  Feature Encoding.

# Using the replace() function to encode categorical data into numerical

# This function takes feature name as a parameter  and returns mapping dictionary to replace(or map) 
# categorical data with numerical data.
def encode_data(feature_name):

    ''' 

    This function takes feature name as a parameter and returns mapping dictionary to replace(or map) categorical data with numerical data.

    '''

    mapping_dict = {}

    unique_values = list(rain[feature_name].unique())

    for idx in range(len(unique_values)):

        mapping_dict[unique_values[idx]] = idx

    return mapping_dict




rain['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)

rain['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)

rain['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)

rain['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)

rain['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)

rain['Location'].replace(encode_data('Location'), inplace = True)

rain.head()


# In[23]:


# Correlations 

plt.figure(figsize=(20,20))
sns.heatmap(rain.corr(), linewidths=0.5, annot=False, fmt=".2f", cmap = 'viridis')


# In[24]:


correlation = rain.corr()
plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
plt.show()


# In[25]:


# MaxTemp,MinTemp,Temp9am,Temp3pm are highly correlated.

# Sunshine, Cloud9am, Cloud3am are also highly correlated.


# In[26]:


sns.pairplot(rain[numerical_features], kind = 'scatter', diag_kind = 'hist', palette = 'Rainbow')
plt.show()


# In[27]:


# For feature importance and feature scaling, we need to split data into independent and dependent features 
X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']


# In[28]:


#  Feature Importance:
 # Machine Learning Model performance depends on features that are used to train a model. 
 # Feature importance describes which features are relevant to build a model.
 # Feature Importance refers to the techniques that assign a score to input/label 
 # features based on how useful they are at predicting a target variable. 
 #Feature importance helps in Feature Selection.
 #We’ll be using ExtraTreesRegressor class for Feature Importance. 
 #This class implements a meta estimator that fits a number of randomized decision trees 
 #on various samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.


# In[30]:


from sklearn.ensemble import ExtraTreesRegressor
etr_model = ExtraTreesRegressor()
etr_model.fit(X,y)
etr_model.feature_importances_


# In[ ]:


feature_imp = pd.Series(etr_model.feature_importances_,index=X.columns)
feature_imp.nlargest(10).plot(kind='barh')


# In[35]:


# Splitting Data into training and testing set
# train_test_split() is a method of model_selection class used to split data into training and testing sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35, random_state = 0)


# In[36]:


# Length of Training and Testing set
print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))


# In[37]:


# Feature Scaling

# Feature Scaling is a technique used to scale, normalize, standardize data in range(0,1). 
# When each column of a dataset has distinct values, then it helps to scale data of each column to a common level. 
# StandardScaler is a class used to implement feature scaling.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[38]:


X_test = scaler.transform(X_test)


# In[ ]:


### -------------------- MODEL BUILDING --------------------------###


# In[ ]:


# Logistic Regression algorithm to build a predictive model to predict whether or not it will rain tomorrow in Australia.
# Logistic Regression: It is a statistic-based algorithm used in classification problems. 
# It allows us to predict the probability of an input belongs to a certain category.


# In[39]:


from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)  # WTF is "random state"?
model = classifier_logreg.fit(X_train, y_train)
model


# In[41]:


# Model Testing

y_pred = model.predict(X_test)


# In[42]:


# # Evaluating Model Performance:
# Accuracy_score() is a method used to calculate the accuracy of a model prediction on unseen data.

from sklearn.metrics import accuracy_score
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)))


# In[43]:


# Checking for Underfitting and Overfitting:

print("Train Data Score: {}".format(classifier_logreg.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_logreg.score(X_test, y_test)))


# In[ ]:


# The accuracy Score of training and testing data is comparable and almost equal. 
# So, there is no question of underfitting and overfitting. And the model is generalizing well for new unseen data.


# In[44]:


# Confusion Matrix:
# A Confusion Matrix is used to summarize the performance of the classification problem. 
# It gives a holistic view of how well the model is performing.


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[47]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[48]:




