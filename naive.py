# Import pandas
import pandas as pd

from utils import plot_data, generate_data
import numpy as np
import matplotlib.pyplot as plt
# Import pandas
import pandas as pd
# For spliting
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel('default of credit card clients.xls', header=1)

# Rename column PAY_0 to PAY_1
df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
# Rename column 
df.rename(columns={'default payment next month':'def_pay'}, inplace=True)

# Seperate X and y from dataset
df_X = df.drop(['def_pay'], axis=1)
df_y = df.def_pay

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)

# Shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.naive_bayes import GaussianNB
# metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error

# GNBayes
gnb_model_1 = GaussianNB()
gnb_model_1.fit(X_train, y_train)
y_pred = gnb_model_1.predict(X_test)

print("#### Naive Bayes with 24 features #### ")
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('\nAccuracy Score for gnb_model_1: ', accuracy_score(y_pred,y_test))


# remove feat
df_Xn = df_X[['LIMIT_BAL', 'EDUCATION', 'AGE']]

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df_Xn, df_y, test_size=0.2, random_state=10)

# Shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# GNBayes with less features
gnb_model_2 = GaussianNB()
gnb_model_2.fit(X_train, y_train)
y_pred = gnb_model_2.predict(X_test)

print("#### Naive Bayes with 3 features #### ")
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('\nAccuracy Score for gnb_model_2: ', accuracy_score(y_pred,y_test))


