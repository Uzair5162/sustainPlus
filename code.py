import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split         #used to split the data to train and test
from sklearn.ensemble import RandomForestClassifier          #used fit fit a random forest


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

algae = pd.read_csv('finalData.csv')

y=algae.alg_con
x_features =algae.drop(["alg_con"],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_features,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_depth = 2, n_estimators=50)

rf_model.fit(x_train, y_train)

print(rf_model.score(x_test,y_test))

feature_importances = pd.DataFrame(rf_model.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)
