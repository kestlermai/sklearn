# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:49:17 2022

@author: 11146
"""

#RF实战调参
#RF通过构建n个决策树进行集成学习，即用不同角度看问题进行投票，输出的类别是不同树输出的类别的众数而定
#随机性体现在每棵树上的训练样本都是随机的，每棵树都是独立的，每个特征也是随机的，随机森林就是为了减少决策树的过拟合
#经典的bagging机器学习

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 666)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#先用默认参数跑
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#结果跟DT一样出现过拟合情况

#那跟DT调参一样，先调整DT的参数
param_grid=[{
            'max_depth':[50,60,70,80,90,100],
            'min_samples_split':[i for i in range(1,11)],
            'min_samples_leaf':[i for i in range(1,11)],
            'max_leaf_nodes':[50,80,100,150,200],           
            },
           ]
#电脑算不过来了，减少参数跟删掉max_leaf_nodes，max_leaf_nodes后面再说
param_grid=[{
            'max_depth':[50,60,70,80,90,100],
            'min_samples_split':[i for i in range(5,11)],
            'min_samples_leaf':[i for i in range(5,11)],         
            },
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#输出最佳参数max_depth=50,min_sample_leaf=7,min_sample_split=5
#测试一下结果如何
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#调整完过拟合就开始调整max_leaf_nodes
#可以看到，max_depth=50、min_samples_split=5，是我们设置范围的最小值
#所以可以跟max_leaf_nodes一起再调整一下，但是范围不能太大：

param_grid=[{
            'max_depth':[20,30,40,50,60],
            'min_samples_split':[i for i in range(2,6)],
            'max_leaf_nodes':[50,80,100,150,200],
},
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
print(grid_search.best_params_)
classifier.fit(X_train, y_train)
#初步得到DT的最优参数max_depth=20, max_leaf_nodes=50,min_sample_leaf=7,
#'min_samples_split=2'这个参数感觉有问题，还是设置为4或者5，后面再看看

#继续调整RF特有的参数：n_estimators、oob_score、criterion
#其中：主要调整n_estimators（决策树的数量）、oob_score、criterion默认就好。
#因为这里只需要调整n_estimators一个参数，所以就连同DT参数一起弄了：
param_grid=[{
            'n_estimators':[i for i in range(10,100,10)],
            'max_depth':[15,20,30],
            'min_samples_split':[3,4,5],
            'min_samples_leaf':[7,8,9],
            'max_leaf_nodes':[30,40,50],           
            },
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#max_depth=15, max_leaf_nodes=30, min_samples_leaf=9,min_samples_split=3, n_estimators=80
#看看最优模型的结果
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

import math
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)   
cm_train = confusion_matrix(y_train, y_trainpred)
#测试集的参数
a = cm[0,0]
b = cm[0,1]
c = cm[1,0]
d = cm[1,1]
acc = (a+d)/(a+b+c+d)
error_rate = 1 - acc
sen = d/(d+c)
sep = a/(a+b)
precision = d/(b+d)
F1 = (2*precision*sen)/(precision+sen)
MCC = (d*a-b*c) / (math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))
auc_test = roc_auc_score(y_test, y_testprba)
#训练集的参数
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train)))
auc_train = roc_auc_score(y_train, y_trainprba)  

#绘画训练集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_trainprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_train), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('ROC of train')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#绘画测试集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_test, y_testprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_test), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')   
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()


#当random_state=428,看看结果
import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 428)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#先用默认参数跑
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

param_grid=[{
            'max_depth':[50,60,70,80,90,100],
            'min_samples_split':[i for i in range(5,11)],
            'min_samples_leaf':[i for i in range(5,11)],         
            },
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#max_depth=50, min_samples_leaf=7, min_samples_split=5

param_grid=[{
            'max_depth':[20,30,40,50,60],
            'min_samples_split':[i for i in range(2,6)],
            'max_leaf_nodes':[50,80,100,150,200],
},
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
print(grid_search.best_params_)
classifier.fit(X_train, y_train)
#max_depth=20, max_leaf_nodes=50, min_samples_split=4，到这一步又好了，可以导出min_samples_split的结果
#调整RF中n_estimators参数
param_grid=[{
            'n_estimators':[i for i in range(10,100,10)],
            'max_depth':[15,20,30],
            'min_samples_split':[3,4,5],
            'min_samples_leaf':[7,8,9],
            'max_leaf_nodes':[30,40,50],           
            },
           ]
boost = RandomForestClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#max_depth=15, max_leaf_nodes=40, min_samples_leaf=7,min_samples_split=3, n_estimators=80,

#看看random_state=428时的最优模型的结果
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

import math
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)   
cm_train = confusion_matrix(y_train, y_trainpred)
#测试集的参数
a = cm[0,0]
b = cm[0,1]
c = cm[1,0]
d = cm[1,1]
acc = (a+d)/(a+b+c+d)
error_rate = 1 - acc
sen = d/(d+c)
sep = a/(a+b)
precision = d/(b+d)
F1 = (2*precision*sen)/(precision+sen)
MCC = (d*a-b*c) / (math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))
auc_test = roc_auc_score(y_test, y_testprba)
#训练集的参数
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train)))
auc_train = roc_auc_score(y_train, y_trainprba)  

#绘画训练集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_trainprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_train), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('ROC of train')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#绘画测试集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_test, y_testprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_test), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')   
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#有提升但不大，看来是我的数据有问题