

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
# from sklearn import cross_validation, metrics   #Additional scklearn functions
pd.set_option('display.max_columns', 500)



df1= pd.read_csv('borrower_table.csv', encoding='latin1', error_bad_lines=False )

df2=pd.read_csv('loan_table.csv', encoding='latin1', error_bad_lines=False )

df1.head()

df1.shape

df2.head()

df2.shape

"""MERGING BOTH DATAFRAMES"""

df1=pd.merge(df1, df2, on=['loan_id'], how = 'inner')

df1.shape

df1.head()

df1.isnull().any()

df1.drop('date', 1, inplace=True)

"""DELETING OUTLIERS"""

down_quantiles = df1['total_credit_card_limit'].quantile(0.05)
df1['total_credit_card_limit'].mask(df1['total_credit_card_limit'] < down_quantiles, down_quantiles,inplace=True)
upper_quantiles = df1['total_credit_card_limit'].quantile(0.95)
df1['total_credit_card_limit'].mask(df1['total_credit_card_limit'] > upper_quantiles, upper_quantiles,inplace=True)

down_quantiles = df1['saving_amount'].quantile(0.05)
df1['saving_amount'].mask(df1['saving_amount'] < down_quantiles, down_quantiles,inplace=True)
upper_quantiles = df1['saving_amount'].quantile(0.95)
df1['saving_amount'].mask(df1['saving_amount'] > upper_quantiles, upper_quantiles,inplace=True)

down_quantiles = df1['checking_amount'].quantile(0.05)
df1['checking_amount'].mask(df1['checking_amount'] < down_quantiles, down_quantiles,inplace=True)
upper_quantiles = df1['checking_amount'].quantile(0.95)
df1['checking_amount'].mask(df1['checking_amount'] > upper_quantiles, upper_quantiles,inplace=True)

down_quantiles = df1['yearly_salary'].quantile(0.05)
df1['yearly_salary'].mask(df1['yearly_salary'] < down_quantiles, down_quantiles,inplace=True)
upper_quantiles = df1['yearly_salary'].quantile(0.95)
df1['yearly_salary'].mask(df1['yearly_salary'] > upper_quantiles, upper_quantiles,inplace=True)

down_quantiles = df1['avg_percentage_credit_card_limit_used_last_year'].quantile(0.05)
df1['avg_percentage_credit_card_limit_used_last_year'].mask(df1['avg_percentage_credit_card_limit_used_last_year'] < down_quantiles, down_quantiles,inplace=True)
upper_quantiles = df1['avg_percentage_credit_card_limit_used_last_year'].quantile(0.95)
df1['avg_percentage_credit_card_limit_used_last_year'].mask(df1['avg_percentage_credit_card_limit_used_last_year'] > upper_quantiles, upper_quantiles,inplace=True)

"""VECTORISING THE TEXT DATA"""

data = pd.get_dummies(df1, columns = ['loan_purpose'])

data.columns

data.shape

data.head(5)

data.to_csv("final_data.csv")

temp = data.drop('loan_granted' , 1)

age=temp['age']
age

import matplotlib.pyplot as plt
import numpy as np
plt.hist(age, density=True, bins=30)
# 3 bins 20-35,35-50,50-80

labels =[1,2,3]
b_age=pd.cut(temp['age'], 3,labels=labels)
b_age
#from the histograms we can verify the age intervals here to be true

temp["age"]=b_age

#df_x=temp.drop("age" , 1)
df_x=temp

df_x.head()

"""REPLACING NUL VALUES WITH -1"""

df_x.fully_repaid_previous_loans.replace(np.nan, -1, regex=True , inplace=True)

df_x.currently_repaying_other_loans.replace(np.nan, -1, regex=True , inplace=True)

df_x.avg_percentage_credit_card_limit_used_last_year.replace(np.nan, -1, regex=True , inplace=True)

df_x.loan_repaid.replace(np.nan, -1, regex=True , inplace=True)

df_x.isnull().any()

df_x.loan_repaid.value_counts()

"""SEPARATING DATA :

1 PEOPLE WHO DIDN'T GET LOAD 

2 PEOPLE WHO GOT LOAN
"""

later_test=df_x.loc[df_x['loan_repaid'] == -1]

later_test.shape

sample_data=df_x.loc[(df_x['loan_repaid'] == 1) | (df_x['loan_repaid'] == 0 )]

sample_data.shape

loan_id= later_test.iloc[:, 0]

loan_id.head()

sample_data= sample_data.drop("loan_id" , 1)

later_test = later_test.drop('loan_id' , 1)

df_y = sample_data.iloc[:, 11]

df_y.head()

df_x = sample_data.drop('loan_repaid' , 1)

df_x.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)

model_1= GradientBoostingClassifier(n_estimators = 300, learning_rate = 0.08, max_depth = 5, random_state = 0)
model_1.fit(x_train, y_train)
print('Train:')
print (300, 0.08, 5, model_1.score(x_train,y_train))
print('Test:')
print (300, 0.08, 5, model_1.score(x_test,y_test))
print('model made')

from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
y_pred = model_1.predict(x_test)

accuracy_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

"""CALCULATING FEATURE IMPORTANCE"""

featureimp = list (zip(df_x.columns, model_1.feature_importances_))
featureimp.sort(key=lambda x: x[1], reverse = True)
featureimp

"""CORRELATION BETWEEN FEATURES 

HIGHER THE SAVING AMOUNT , MORE LIKELY THE LOAN IS REPAID.
"""

sample_data.corr()

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

actual_x_pred= later_test.drop('loan_repaid' , 1)

actual_y_pred = model_1.predict(actual_x_pred)

actual_y_pred

later_test['pred_loan_repaid'] = actual_y_pred

later_test.head()

loan_id.head()

later_test = pd.merge(later_test,loan_id, left_index=True, right_index=True)

later_test.head()

later_test.to_csv("pred_final_data.csv")

final_cust=later_test['loan_id'].loc[(later_test['pred_loan_repaid'] == 1.0)]

final_cust=pd.DataFrame(list(final_cust))
final_cust.to_csv("final_loan_id.csv")

