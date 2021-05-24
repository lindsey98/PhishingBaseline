

import xgboost as xgb
import _pickle as pickle
import pandas as pd
import joblib
from sklearn.model_selection import KFold,train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score
from sklearn.datasets import load_iris,load_digits,load_boston
import numpy
import matplotlib.pyplot as plt

##data load
with open('FGCS_benign_train.pkl','rb') as handle:
    benign_train = pickle.load(handle)
# print(len(benign_train))

##data load
with open('FGCS_benign_test.pkl','rb') as handle:
    benign_test = pickle.load(handle)
# print(len(benign_test))

##data load
with open('FGCS_phish_train.pkl','rb') as handle:
    phish_train = pickle.load(handle)
# print(len(phish_train))

##data load
with open('FGCS_phish_test.pkl','rb') as handle:
    phish_test = pickle.load(handle)
# print(len(phish_test))



train = pd.concat([benign_train, phish_train], axis = 0)
y_train = train.phish
x_train = train.drop(columns = ['phish'])

val = pd.concat([benign_test, phish_test], axis = 0)
y_val = val.phish
x_val = val.drop(columns = ['phish'])


print(len(x_train))
print(list(y_train).count(1))
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)
joblib.dump(xgb_model,'model.pkl')

y_pred = xgb_model.predict(x_val)
print(confusion_matrix(y_val, y_pred))
print(accuracy_score(y_val, y_pred))





