# Targeting_Potential_Customers

Problem Description:
A website sent advertisements by emails to users who interested in their product. The task is to find a good model to predict if an advertisement will be clicked. 

Tasks:
1.Explore the basic information of the datasets
2.Clean and preprocess the datasets (such as missing values, outliers, dummy, merging etc.)
3.Model Generation and Evaluation. 

Dataset Files (Total three files)
user_features.csv  | features describing our users
product_feature.csv  | features describing products shown in the advertisements. 
click_history.csv  | contains which products users had previously seen and whether that user ordered products in this website before.

Detail of code:
Dataset is being explored,cleaned and prprocessed, finally three csv files are joined to build five diffeernt models
1) Random Forests
2) Neural Networks
3) Support Vector Machine
4) Logistic Regression
5) Ada Boost

Full working code in Python with detail comments and expalnation is uploaded. Finally best chosen model was improved using SMOTE and ROC curve is produced.
The real challenge in the dataset is to tranfom list values like ['tools', 'makeup', 'foot', 'nail'] to dummy variables.
Provided code has full description about each step.



