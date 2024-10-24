#Grid search cross validation, Random search and Roc curve

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset
data=pd.read_csv(r"C:\Users\arati\DATAS SCIENCE NIT\OCTOBER\Social_Network_Ads.csv")
 
data.head()

# dipendent andindependent variable
x=data.iloc[:,[2,3]].values
y=data.iloc[:,-1].values

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)


# standard scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# model building 
# 1. SVM->SVC
from sklearn.neighbors import KNeighborsClassifier
clasifier=KNeighborsClassifier()
clasifier.fit(x_train,y_train)

# prediction
y_pred=clasifier.predict(x_test)
#print(y_pred)

# model accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print("model accuracy:- ",ac)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("***confusion matrix***")
print(cm)

# bais
bais=clasifier.score(x_train,y_train)
print("Bais:- ",bais)

#variance
var=clasifier.score(x_test,y_test)
print("Variance;- ",var)

#---------------------------------------------
# Grid search CV KNN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Hyperparameters
param_grid_knn = {
    "n_neighbors":[1,2,3,4,5,6,7,8,9,10],
    'weights':['uniform', 'distance'],
    'leaf_size': [10,20,30,40,50],
    'metric': ['euclidean', 'manhattan']
}

#impliment gridsearchcv
grid_knn = GridSearchCV(clasifier, param_grid_knn, cv=5)
grid_knn.fit(x_train, y_train)

# find best parameter
print("Best parameters GridSearchCV:", grid_knn.best_params_)

# prediction
y_pred_knn_grid = grid_knn.predict(x_test)

# accuracy score
print("Accuracy GridSearchCV: ", accuracy_score(y_test, y_pred_knn_grid))

# ------------------------------------------------------
# Random search cv
#Hyperparameters
param_random_knn = {
    "n_neighbors":[1,2,3,4,5,6,7,8,9,10],
    'weights':['uniform', 'distance'],
    'leaf_size': [10,20,30,40,50],
    'metric': ['euclidean', 'manhattan']
}

random_knn = RandomizedSearchCV(clasifier, param_random_knn, n_iter=10, cv=5)
random_knn.fit(x_train,y_train)

print("Best parameters in random search CV:-", random_knn.best_params_)

y_pred_knn_random=random_knn.predict(x_test)

print("accuracy score of Random search CV:- ",accuracy_score(y_test,y_pred_knn_random))

#-----------------------------------------------------
# auc roc curve
from sklearn.metrics import auc,roc_curve,roc_auc_score
# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.2f}")

#Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
