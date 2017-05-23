import base
from base import load_2year

from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import AdaBoostClassifier

bank2data=load_2year()
X=bank2data[0]
y=bank2data[1]
clf=AdaBoostClassifier(n_estimators=100)

scores=cross_val_score(clf,X,y,scoring='roc_auc')

print(scores)
