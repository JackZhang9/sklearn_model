# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/14 14:50
'''
# 导入相关库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1.iris
# 建模SVM实例
# 导入数据集
data_iris=pd.read_csv(r'G:\sklearn_model\sklearn数据集\分类\iris.csv')
print(data_iris.head())

# 划分特征数据和label数据
data_x=pd.DataFrame(data_iris,columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
data_y=pd.DataFrame(data_iris,columns=['label'])
print(data_x.head(),data_y.head())

# 划分训练数据和测试数据
x_train, x_test, y_train, y_test=train_test_split(data_x,data_y,test_size=0.33,random_state=916)
print(len(x_train),len(y_train))

# 数据标准化
stdscale=StandardScaler()
x_train_stded=stdscale.fit_transform(x_train)
x_test_stded=stdscale.fit_transform(x_test)

# 数据归一化
Mscaler=MinMaxScaler()
x_train_Mscaler=Mscaler.fit_transform(x_train_stded)
x_test_Mscaler=Mscaler.fit_transform(x_test_stded)


# *****************************************************************************************
# 建模SVM实例
# 使用网格搜索，搜素参数n
svm_paras=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}]
# 默认的初始化模型
svc_iris=SVC()

# 网格搜索实例化
grid_iris=GridSearchCV(svc_iris,svm_paras,cv=10)

# 训练模型
yy_train=np.ravel(y_train)   # 把y_train变成ndarray类型，在sklearn训练时，数据都要是ndarray格式
grid_iris.fit(x_train_Mscaler,yy_train)
print('grid_iris.best_estimator_={},grid_iris.best_params={},grid_iris.best_score_={}'.format(grid_iris.best_estimator_,grid_iris.best_params_,grid_iris.best_score_))
# grid_iris.best_estimator_=SVC(C=1, kernel='linear'),
# grid_iris.best_params={'C': 1, 'kernel': 'linear'},
# grid_iris.best_score_=0.9800000000000001

# 基于搜索的paras结果重新建模
svc_iris1=SVC(C=1,kernel='linear')
svc_iris1.fit(x_train_Mscaler,yy_train)

# 测试集预测
iris_predic=svc_iris1.predict(x_test_Mscaler)
yy_test=np.ravel(y_test)
print([iris_predic-yy_test])
# [array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  1,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,
#         0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#       dtype=int64)]

# 准确率
accuracy=accuracy_score(yy_test,iris_predic)
print('accuracy={}'.format(accuracy))
# accuracy=0.92























