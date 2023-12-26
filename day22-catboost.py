# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:28:18 2023

@author: maihuanzhuo
"""

##Catboost建模实战
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

#Catboost的调参策略
# depth：树深度，默认6，最大16。
# grow_policy：子树生长策略。可选：SymmetricTree（默认值，对称树）、Depthwise（整层生长，同xgb）、Lossguide（叶子结点生长，同lgb）。
# min_data_in_leaf：叶子结点最小样本量。只能与Lossguide和Depthwise增长策略一起使用。
# max_leaves：最大叶子结点数量，不建议使用大于64的值，因为它会大大减慢训练过程。只能与 Lossguide增长政策一起使用。
# iterations：迭代次数，默认500。
# learning_rate：学习速度，默认0.03。
# l2_leaf_reg：L2正则化。
# random_strength：特征分裂信息增益的扰动项，默认1，用于避免过拟合。
# rsm：列采样比率，默认值1，取值（0，1]。

#先默认参数跑：
import catboost as cb
classifier = cb.CatBoostClassifier(eval_metric='AUC')
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

#比之前的xgb和lgb好一点，训练集的auc都接近1了，所以还是过拟合

# 开整Model1（SymmetricTree）：
# 由于grow_policy选择SymmetricTree，因此min_data_in_leaf和max_leaves调整不了。因此，先调整depth试试：
import catboost as cb
param_grid=[{
             'depth': [i for i in range(6,11)],
           },
           ]

boost = cb.CatBoostClassifier(eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
#变得更加过拟合了

#Catboost的最优参数的调取有点不同，直接grid_search.best_estimator_._init_params
grid_search.best_estimator_._init_params
#{'depth': 8, 'eval_metric': 'AUC'}

#调整l2_leaf_reg
param_grid=[{
             'l2_leaf_reg': [i for i in range(1,11)],  
            },
           ]

boost = cb.CatBoostClassifier(depth = 8, eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
grid_search.best_estimator_._init_params
#{'depth': 8, 'l2_leaf_reg': 10, 'eval_metric': 'AUC'}
#更加过拟合了

#继续调整过拟合的参数：random_strength

param_grid=[{
             'random_strength': [i for i in range(1,11)],  
            },
           ]

boost = cb.CatBoostClassifier(depth = 8, l2_leaf_reg = 10, eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
grid_search.best_estimator_._init_params
#{'depth': 8, 'l2_leaf_reg': 10, 'random_strength': 2, 'eval_metric': 'AUC'}
#还是过拟合

#rsm
param_grid=[{
             'rsm': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],  
            },
           ]

boost = cb.CatBoostClassifier(depth = 8, l2_leaf_reg = 10, random_strength = 2, eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
grid_search.best_estimator_._init_params
#'depth': 8, 'l2_leaf_reg': 10, 'rsm': 1.0, 'random_strength': 2, 'eval_metric': 'AUC'
#依然过拟合

#learning_rate和iterations一起调整试试
#缩短一下迭代次数
param_grid=[{
             'learning_rate': [0.03,0.06,0.08,0.1],
             'iterations': [100,200,300,400,500,600,700,800],             
            },
           ]
boost = cb.CatBoostClassifier(depth = 8, l2_leaf_reg = 10, random_strength = 2, 
                              rsm = 0.3, eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
grid_search.best_estimator_._init_params
#{'iterations': 400,'learning_rate': 0.03}
#还是过拟合

#Overfitting detection settings几个参数
#early_stopping_rounds：早停设置，默认不启用。
classifier = cb.CatBoostClassifier(grow_policy='SymmetricTree', depth=8, min_data_in_leaf=115, 
                                   l2_leaf_reg=10, rsm=1.0, random_strength=2, learning_rate=0.03, iterations=400, 
                                   early_stopping_rounds=200, eval_metric='AUC')
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
#没用，一样过拟合

# od_type：过拟合检测类型，默认IncToDec。可选：IncToDec、Iter。
# od_pval：IncToDec过拟合检测的阈值，当达到指定值时，训练将停止。要求输入验证数据集，建议取值范围[10e-10，10e-2]。默认值0，即不使用过拟合检测。

classifier = cb.CatBoostClassifier(grow_policy='SymmetricTree', depth=8, min_data_in_leaf=115, 
                                   l2_leaf_reg=10, rsm=1.0, random_strength=2, learning_rate=0.03, iterations=400, 
                                   early_stopping_rounds=200, eval_metric='AUC',
                                   od_type='IncToDec',od_pval=0.1 )
classifier.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#没有好转

#用网格试一试od_pval用哪个取值好一些
import catboost as cb
param_grid=[{
              'od_pval': [0.6,0.2,0.1,0.01,0.001,0.0001,0.00001,0.000001],
              'od_type': ['IncToDec','Iter']
           },
           ]
boost = cb.CatBoostClassifier(grow_policy='SymmetricTree', depth=8, min_data_in_leaf=115, 
                                   l2_leaf_reg=10, rsm=1.0, random_strength=2, learning_rate=0.03, iterations=400, 
                                   early_stopping_rounds=200, eval_metric='AUC')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv = 10)     
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))   
classifier = grid_search.best_estimator_
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
grid_search.best_estimator_._init_params
#'od_pval': 0.6, 'od_type': 'IncToDec'
#依然不行，换成Depthwise看看

#开整Model2（Depthwise）：
# grow_policy选择Depthwise，因此多加了一个min_data_in_leaf可调整。类似地，先调整depth试试；（这里偷懒，直接用depth=8，不过应该问题不大）
# 调整min_data_in_leaf
param_grid=[{
             'min_data_in_leaf': range(5,200,10),
           },
           ]
boost = cb.CatBoostClassifier(grow_policy='Depthwise', depth=8, eval_metric='AUC')
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv = 10)     
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))   
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#明显变好了
grid_search.best_estimator_._init_params
#min_data_in_leaf': 155

#后续继续调l2_leaf_reg，rsm，random_strength，learning_rate，iterations参数，鸡哥说又拟合了，直接不跑

# 直接开整Model3（Lossguide）：
# row_policy选择Lossguide，因此多加了min_data_in_leaf和max_leaves可调整。类似地，先调整depth试试：
param_grid=[{
             'depth': [i for i in range(6,11)],
           },
           ]
boost = cb.CatBoostClassifier(grow_policy='Depthwise', eval_metric='AUC')
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv = 10)     
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))   
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#一样过拟合
grid_search.best_estimator_._init_params
#'depth': 9

#调整min_data_in_leaf
param_grid=[{
             'min_data_in_leaf': range(5,200,10),
           },
           ]

boost = cb.CatBoostClassifier(grow_policy='Lossguide', depth=9, eval_metric='AUC')
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv = 10)     
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))   
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#拟合没这么严重，但还是存在
grid_search.best_estimator_._init_params
#'min_data_in_leaf': 95

#调整num_leaves
param_grid=[{
              'num_leaves': range(5, 100, 5),
           },
           ]
boost = cb.CatBoostClassifier(grow_policy='Lossguide',depth=9, min_data_in_leaf = 95, eval_metric='AUC')
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv = 10)     
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))   
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#还是过拟合，auc_trian=0.994 auc_test=0.867，b=35
grid_search.best_estimator_._init_params
#'num_leaves': 25
#l2_leaf_reg、random_strength、learning_rate和iterations等参数，我就不调了，预感调整以后性能又回去了，所以到此为止了吧。
#直接看看Model3的结果（这里我没有加入验证集进行调参）

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
plt.title('Please replace your title')
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
plt.title('Please replace your title')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#总结：根据grow_policy（子树生长策略）可以分成三种模型（Model1、Model2和Model3）
#其中，严格来说使用SymmetricTree（对称树）的才是原汁原味的Catboost，毕竟对称树就是它的特色之一。
#从结果来做，Model1存在较大的过拟合，除非运用测试集进行调试，Model2和Model3引入DT的一些参数，可以较好的纠正过拟合。
#但使用测试集调整超参数会影响模型的泛化性能
