# Import pandas
import pandas as pd
# For spliting
from sklearn.model_selection import train_test_split
# Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error


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


# Logistic Regression with l1 penalty
log_model1 = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)
log_model1.fit(X_train, y_train)

y_pred = log_model1.predict(X_test)

print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('\nAccuracy Score for log_model_1: ', accuracy_score(y_pred,y_test))

# Logistic Regression with l2 penalty
log_model2 = LogisticRegression(penalty='l2',solver='liblinear',max_iter=1000)
log_model2.fit(X_train, y_train)

y_pred = log_model2.predict(X_test)

print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('\nAccuracy Score for log_model_2: ', accuracy_score(y_pred,y_test))


# Import sk Neural network
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(hidden_layer_sizes=(50,),max_iter=200, tol=0.000000001, early_stopping=False, validation_fraction=0.2, n_iter_no_change=200)
# Train
NN.fit(X_train, y_train)
# Test
y_pred = NN.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)

# Printing the Results
print("#### Neural Network with 1 layer of 50 nodes #### ")
print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix: ")
print(confusion_mat)
print("Classification Report: ")
print(classification_report(y_pred, y_test))

import matplotlib.pyplot as plt
plt.plot([epoch for epoch in range(200)], NN.loss_curve_, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("NN1.png")


NN2 = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, tol=0.000000001, early_stopping=False, validation_fraction=0.2, n_iter_no_change=200)
# Train
NN2.fit(X_train, y_train)
# Test
y_pred = NN2.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)

# Printing the Results
print("#### Neural Network with 2 layer of 50 nodes each #### ")
print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix: ")
print(confusion_mat)
print("Classification Report: ")
print(classification_report(y_pred, y_test))


plt.plot([epoch for epoch in range(200)], NN2.loss_curve_, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("NN2.png")

NN3 = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=1000, tol=0.000000001, early_stopping=False, validation_fraction=0.2, n_iter_no_change=200)
# Train
NN3.fit(X_train, y_train)
# Test
y_pred = NN3.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)

# Printing the Results
print("#### Neural Network with 3 layer of 50 nodes each #### ")
print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix: ")
print(confusion_mat)
print("Classification Report: ")
print(classification_report(y_pred, y_test))

plt.plot([epoch for epoch in range(200)], NN3.loss_curve_, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("NN3.png")


# Compare NN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

common_params = {
    "X": df_X,
    "y": df_y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([NN, log_model2]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")


    
    
# GNBayes
gnb_model1 = GaussianNB()
gnb_model1.fit(X_train, y_train)
y_pred = model_gnb.predict(X_test)
print('\nAccuracy Score for model1: ', accuracy_score(y_pred,y_test))

# Compare GN
# Compare 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

common_params = {
    "X": df_X,
    "y": df_y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([gnb_model1, log_model2]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
    
    