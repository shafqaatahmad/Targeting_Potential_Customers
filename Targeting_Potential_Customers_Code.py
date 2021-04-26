# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:07:23 2021

@author: shafqaat.ahmad
"""

import pandas as pd
import numpy as np
import os 
import seaborn as sns
import matplotlib.pylab as plt
from numpy import nan
import re
from sklearn.preprocessing import StandardScaler,LabelEncoder
import seaborn as sns
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score,cross_validate,cross_val_predict
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from ann_visualizer.visualize import ann_viz;
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




## Load data
path="E:\Python WD" ## Use your own directory
os.chdir(path) 
# Loading the file and assigning to datafarme user_features
user_features = pd.read_csv("user_features.csv") #Loading user features
product_features = pd.read_csv("product_features.csv") #Loading Product features
click_history = pd.read_csv("click_history.csv")  #Loading Click history

## We will first explore each file, clean and preprocess, finally we will merge files
## and run them through models ideally Random Forest and AdaBoost

##### Working on user_features dataset

user_features.info() # get general info about the dataset
user_features.describe() # dataframe description
#Clean
user_features.isnull().sum() # check for missing values

## There are 500 null values in number_of_clicks_before
## imputing them with Zero, assumption ... .
user_features["number_of_clicks_before"].fillna(0,inplace=True)
user_features.isnull().sum() # check for missing values

user_features.duplicated().sum() # check for duplicates 

user_features.dtypes #check the datatypes

# Converting numeric_of_clicks_before to numeric

try:
 pd.to_numeric(user_features["number_of_clicks_before"])
except:
 print("An exception occurred")
finally:
 pass


# Here we need to replace 6+ with some integer. 6+ means any number
# greater than 6. We are replacing 6+ with 7, value 7 will differntiate
# 6+ from remaining values, and that is what we need over here i.e
# draw a clear boundry between 6+ values and rest of values.  

user_features["number_of_clicks_before"]=user_features["number_of_clicks_before"].replace("6+","7")

# Converting numeric_of_clicks_before to numeric
user_features["number_of_clicks_before"]=pd.to_numeric(user_features["number_of_clicks_before"])

# Checking numeric_of_clicks_before values
user_features.number_of_clicks_before.value_counts()
#check the datatypes
user_features.dtypes 


# Column "personal_interests" has list of values. We have to modify
# this column as we cannot go along with list columns. The ideal
# solution over here is to convert this column into dummy variables.
# We believe that this will suffice over requirement

# Before converting  "personal_interests" column to dummy variables
# we want to check if there are any duplicates in list. 
# For example in the list ['tools', 'makeup', 'foot', 'nail'.'tools']
# tools is dupliacted. If that is the case then it means that row has
# wrong list values. It will create problems and mislead us while creating dummy
# variables. We will drop such row beacsue we assume that this situation happens becasue
# data was not captured properly or data has been infiltrated.

# Checking dupliactes in list values
i=0 # setting counter
for i in (user_features.index): # looping trough dataset rows 
    # splitting each row personal_interests column and assigining it listsplit 
    listsplit=user_features.iloc[i,3].split(",") 
    # Iterating through list listsplit variable and count those values which apperas 
    # more than once. Finally sum count of those which appear more than once
    listsum=sum([listsplit.count(x) > 1 for x in listsplit])
    #If listsum count is more than zero than it is an indication of duplicate value
    if listsum>0:
       # Print user_id value, so that we can remove that row
       print(user_features.iloc[i,0])

# Nothing printed, so no dplicates in list

# Now the next step is to convert "personal_interests" to dummy variables
# However, the conversion is not that straigh forward. We have to first split list
# then join back it with some operator. Finally we have to remove "[]" and " ' "
# characters from string

# Below in one line of code we are perfoming multiple operations.
# First splitting str on the basis of "," and joining back using "|" opeartor. Reason being
# it is a list and we have to treat it like string. If we call join without splitting
# it will joing characters, not string. 
# Then removing " ][ " and "'"  and all spaces from final string so that
# it can be converetd to dummy variables. Removing spaces is important otherwise
# dummy function will create multiple columns with same name. 
# A point to be noted here, there are some rows which have empty list avlues,
# calling dummy function will add zero value in all variables automatically.

personal_interests_dummies=user_features["personal_interests"].str.split(",").str.join('|').str.replace(r"[]['\s+]","").str.get_dummies()

# Joining back to reminiang columns

user_features=user_features.join(personal_interests_dummies)
#dropping "personal_interests" column since it is changed to dummy 
user_features=user_features.drop("personal_interests",axis=1)


# Beacuse some algorithm only accepts numeric values so we are converting
# "ordered_before" column to 1/0 instaed of True/False
user_features["ordered_before"]=user_features["ordered_before"].astype(int)

user_features.dtypes

user_features[["number_of_clicks_before","Ordered_before"]].hist(figsize=(10,10),color='g')
user_features[["number_of_clicks_before","Ordered_before"]].boxplot()







#### Working on Product_features dataset

product_features.info() # get general info about the dataset
product_features.describe() # dataframe description
product_features.isnull().sum() # check for missing values
product_features.duplicated().sum() # check for duplicates 

product_features.dtypes #check the datatypes

# There are no null values and no duplicates, data type of
# columns are also fine.The "avg_review_score" column is float
# and contains negative values. We dont have any knowledge about 
# how data was collecetd and if negative is a legit value here.
# We assume that "avg_review_score" can have negative values.


# Beacuse some algorithm only accepts numeric values so we are converting
# "on_sale" column to 1/0 instaed of True/False and applying LabelEncoding on
# "category" column to cnvert it to numeric categories

product_features["on_sale"]=product_features["on_sale"].astype(int)

#Applying labelencoding
label_encoder = preprocessing.LabelEncoder()
product_features["category"]=label_encoder.fit_transform(product_features["category"])

product_features[["category","number_of_reviews","avg_review_score"]].hist(figsize=(10,10),color='y')

# Plotting box plot

product_features[["category","number_of_reviews","avg_review_score"]].boxplot()

# Column "number_of_reviews" have some extereme outliers we need to discard them
# We are using IQR method for identifying outliers 

# identify outliers in number_of_reviews
Q1 =product_features[["number_of_reviews"]].quantile(0.25)
Q3 =product_features[["number_of_reviews"]].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = float(Q1-(1.5*IQR))
Upper_Whisker = float(Q3+(1.5*IQR))

print(Upper_Whisker)

# We will use only upper limit to eliminate values, becasue there is no value less
# than lower limit

product_features=product_features[~(product_features["number_of_reviews"]>Upper_Whisker)]

product_features[["category","number_of_reviews","avg_review_score"]].hist(figsize=(10,10),color='y')

product_features[["category","number_of_reviews","avg_review_score"]].boxplot()

product_features.dtypes


#### Working on click_history dataset
# click_history dataset is labeled dataset. We will join
# this dataset with other two datasets to create final dataset.


click_history.info() # get general info about the dataset
click_history.describe() # dataframe description
click_history.isnull().sum() # check for missing values
click_history.duplicated().sum() # check for duplicates 

click_history.dtypes #check the datatypes

# Beacuse some algorithm only accepts numeric values so we are converting
# "click_history" column to 1/0 instaed of True/False and applying LabelEncoding on
# "category" column to cnvert it to numeric categories

click_history["clicked"]=click_history["clicked"].astype(int)



### Now we are ready for joing three dataset
## user_features, product_features  and click_history

# Joining user_features and click_history on common key user_id

join_user_clicks=pd.merge(user_features,click_history,on='user_id')
final_ds=pd.merge(join_user_clicks,product_features,on='product_id')
final_ds=final_ds.drop(columns=["user_id", "product_id"])
final_ds.dtypes

corr = final_ds[['number_of_clicks_before','number_of_reviews','avg_review_score']].corr()
sns.heatmap(corr, annot=True)
################################################
###############Model Building###################

col = final_ds.pop("clicked")
final_ds.insert(0, col.name, col)

final_ds=shuffle(final_ds)


#sm = SMOTE(random_state=42)
#X_sm, y_sm = sm.fit_resample(X, y)


X= final_ds.iloc[:, 1 : 18]
y =final_ds.clicked

# Although scaling is only required for Neural Neural Network. 
# Since, We are fitting five models so we will use "Scaled" data
# for all models in order to properly compare their accuracy performances.
 
#Scaling data
scaler = MinMaxScaler()

final_ds_scaled = scaler.fit_transform(final_ds)
final_ds_scaled = pd.DataFrame(final_ds_scaled, columns = final_ds.columns)

X_scaled = final_ds_scaled.iloc[:, 1 : 18]
y_scaled = final_ds_scaled.clicked


#X_train_scaled=scaler.fit_transform(X_train)
#y_train_scaled=scaler.fit_transform(y_train)

#X_train, X_test, y_train, y_test = train_test_split(
#      X_scaled, y_scaled, test_size=0.3, random_state=679)


# We will use cross validation for all algorithms using RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, random_state=321,n_repeats=2)


# Hyper parameter tuning for Random Forest using Grid Search
# Below three parameters are most important so we are tuning them

# Note: Please note that hyper parameter tuning takes hours , so it is
# not feasible for us to use large list of values or many parameters.


#Model 1 Random Forest

#parameters = {'max_depth': range(2, 20),
#'min_samples_leaf': np.linspace(0.05, 0.4, 10)}

param_grid_rf = {
    'max_depth': range(20, 21),
    'min_samples_split': ((np.linspace(0.0001, 0.001, 5))),
    'n_estimators': range(120,121),
            }


rf = RandomForestClassifier(random_state = 123)

rf_optimizer = GridSearchCV(estimator = rf, param_grid = param_grid_rf,return_train_score=True, 
                          cv = cv, n_jobs = -1, verbose = True,scoring="accuracy")


rf_optimizer.fit(X_scaled, y_scaled)

rf_optimizer.best_params_

rf_optimizer.best_score_

results_rf = pd.DataFrame(rf_optimizer.cv_results_)[['mean_test_score','mean_train_score']].mean().round(2)

Final_Models_Metrics=pd.DataFrame(columns=['Model','Mean_test_score','Mean_train_score','Best_Parameters'])

Final_Models_Metrics=Final_Models_Metrics.append({'Model': 'Random Forest', 
        'Mean_test_score': results_rf.mean_test_score, 'Mean_train_score': 
            results_rf.mean_train_score,'Best_Parameters': 
            rf_optimizer.best_params_}, ignore_index=True)


#Model 2 Neural Network

param_grid_nn = {
    'hidden_layer_sizes': [12,15,18,21,24,27,30,33,36],
    'activation': ['tanh', 'relu','logistic']
                  }


nn = MLPClassifier(max_iter=10000, random_state = 841)

nn_optimizer = GridSearchCV(nn, param_grid_nn, 
                            scoring = "accuracy",
                            cv = cv,
                            return_train_score=True, 
                            verbose = 1,n_jobs=-1)


nn_optimizer.fit(X_scaled, y_scaled)

nn_optimizer.best_params_

results_nn = pd.DataFrame(nn_optimizer.cv_results_)[['mean_test_score','mean_train_score']].mean().round(2)

Final_Models_Metrics=Final_Models_Metrics.append({'Model': 'Neural Networks', 
        'Mean_test_score': results_nn.mean_test_score, 'Mean_train_score': 
            results_nn.mean_train_score,'Best_Parameters': 
           nn_optimizer.best_params_}, ignore_index=True)


#Model 3 AdaBoost

param_grid_ada = {
    'n_estimators': range(80,150,10), 
    'learning_rate': [0.001,0.005,0.01]
                 }

ada = AdaBoostClassifier(random_state = 21)

ada_optimizer = GridSearchCV(ada, param_grid = param_grid_ada,return_train_score=True, 
                          cv = cv, n_jobs = -1, verbose = True,scoring="accuracy")


ada_optimizer.fit(X_scaled, y_scaled)

ada_optimizer.best_params_


results_ada = pd.DataFrame(ada_optimizer.cv_results_)[['mean_test_score','mean_train_score']].mean().round(2)

Final_Models_Metrics=Final_Models_Metrics.append({'Model': 'Ada Boost', 
        'Mean_test_score': results_ada.mean_test_score, 'Mean_train_score': 
            results_ada.mean_train_score,'Best_Parameters': 
           ada_optimizer.best_params_}, ignore_index=True)

#Model 4 SVM

param_grid_svm = {
    'kernel': ['linear','poly','rbf'], 
    'gamma':  [0.2],#0.01,0.05,0.1]
    'degree': [2]#[2,3]         
                }

svm = SVC(random_state = 21)

svm_optimizer = GridSearchCV(estimator = svm, param_grid = param_grid_svm,return_train_score=True, 
                          cv = cv, n_jobs = -1, verbose = True,scoring="accuracy")


svm_optimizer.fit(X_scaled, y_scaled)

svm_optimizer.best_params_


results_svm = pd.DataFrame(svm_optimizer.cv_results_)[['mean_test_score','mean_train_score']].mean().round(2)

Final_Models_Metrics=Final_Models_Metrics.append({'Model': 'SVM', 
        'Mean_test_score': results_svm.mean_test_score, 'Mean_train_score': 
            results_svm.mean_train_score,'Best_Parameters': 
           svm_optimizer.best_params_}, ignore_index=True)


#Model 5 Logistic regression

param_grid_lr = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'], 
    'C': [5.0,1.0, 0.05, 0.01],
                }

lr = LogisticRegression(random_state = 21)

lr_optimizer = GridSearchCV(estimator = lr, param_grid = param_grid_lr ,return_train_score=True, 
                          cv = cv, n_jobs = -1, verbose = True,scoring="accuracy")


lr_optimizer.fit(X_scaled, y_scaled)

lr_optimizer.best_params_


results_lr = pd.DataFrame(lr_optimizer.cv_results_)[['mean_test_score','mean_train_score']].mean().round(2)

Final_Models_Metrics=Final_Models_Metrics.append({'Model': 'Logistic Regression', 
        'Mean_test_score': results_lr.mean_test_score, 'Mean_train_score': 
            results_lr.mean_train_score,'Best_Parameters': 
           lr_optimizer.best_params_}, ignore_index=True)


x2=sns.barplot(x="Model", y="Mean_train_score", hue="Model",data=Final_Models_Metrics)
x2.set(xlabel="Train Metrics")

x1=sns.barplot(x="Model", y="Mean_test_score", hue="Model",data=Final_Models_Metrics)
x1.set(xlabel="Test Metrics")



#### CLassification Report and ROC Curve###

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_scaled, y_scaled)

X_train, X_test, y_train, y_test = train_test_split(
      X_sm, y_sm, test_size=0.3, random_state=679)

nn_model= MLPClassifier(max_iter=10000, random_state = 841,activation= 'relu',
                        hidden_layer_sizes= 24)

nn_model.fit(X_train,y_train)
nn_pred = nn_model.predict(X_test)
metrics.accuracy_score(nn_pred, y_test)
print(metrics.classification_report(y_test, nn_pred))


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
nn_probs = nn_model.predict_proba(X_test)
# keep probabilities for the positive outcome only
nn_probs = nn_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
nn_auc = roc_auc_score(y_test, nn_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Neural Newtork AUC: ROC AUC=%.3f' % (nn_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(nn_fpr, nn_tpr, marker='.', label='Neural Network')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
