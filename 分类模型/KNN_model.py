# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/14 14:50
'''
# 导入相关库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score

# KNN分类问题，
# 1.iris

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

# 建模KNN实例
knn_iris=KNeighborsClassifier(n_neighbors=5)

# 训练模型
yy_train=np.ravel(y_train)   # 把y_train变成ndarray类型，在sklearn训练时，数据都要是ndarray格式
knn_iris.fit(x_train_Mscaler,yy_train)

# 测试集预测
iris_predic=knn_iris.predict(x_test_Mscaler)
yy_test=np.ravel(y_test)
print([iris_predic-yy_test])

# 准确率
accuracy=accuracy_score(yy_test,iris_predic)
print('accuracy={}'.format(accuracy))

# 至此一个简单knn模型建立完毕，但还有很多瑕疵，比如n的设置，是否达到最优
# 这个流程主要是为了说明使用sklearn框架建立knn模型的基本步骤







