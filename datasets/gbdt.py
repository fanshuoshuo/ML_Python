import base
from base import load_2year

from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

bank2data=load_2year()
X=bank2data[0]
y=bank2data[1]
clf=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,
                               max_depth=1,random_state=0)

scores=cross_val_score(clf,X,y,scoring='roc_auc')

print(scores)
