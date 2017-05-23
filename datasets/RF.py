import base
from base import load_2year

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
bank2data=load_2year()
X=bank2data[0]
y=bank2data[1]
"""
clf=RandomForestClassifier(n_estimators=10)
#scores=cross_val_score(clf,X,y,cv=5,scoring='f1_macro')

#print(scores)

#scores=cross_val_score=(clf,X,y,cv=5,scoring='precision')
scores=cross_val_score(clf,X,y,cv=5,scoring='precision')
print(scores)

scores=cross_val_score(clf,X,y,cv=5,scoring='recall')
print(scores)

scores=cross_val_score(clf,X,y,cv=5,scoring='roc_auc')
print(scores)


clf=svm.SVC()
#scores=cross_val_score(clf,X,y,cv=5,scoring='f1_macro')

#print(scores)

#scores=cross_val_score=(clf,X,y,cv=5,scoring='precision')
scores=cross_val_score(clf,X,y,cv=5,scoring='precision')
print(scores)

scores=cross_val_score(clf,X,y,cv=5,scoring='recall')
print(scores)

scores=cross_val_score(clf,X,y,cv=5,scoring='roc_auc')
print(scores)
"""
clf=ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,X,y,cv=5,scoring='roc_auc')
print(scores)
