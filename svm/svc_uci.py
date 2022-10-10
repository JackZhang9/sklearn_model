import pandas as pd
from sklearn.svm import SVC
import pickle

pd.set_option('display.max_rows', None, 'display.max_columns', None)

#导入数据
data=pd.read_csv(r'G:\kaggle_datas\UCI_Credit_Card\UCI_Credit_Card.csv')

print(data.head())
# print(data.isnull().sum())  #无缺失值


#得到标签数据
y = data['default.payment.next.month']

#Load X Variables into a Pandas Dataframe with columns
X = data.drop(['ID','default.payment.next.month'], axis = 1)


# print(y.head())
# print(X.head())

# 导入svc模型
svc_model=SVC(gamma='auto')
svc_model.fit(X,y)
# 保存模型
pickle.dump(svc_model,open('svc_model.pkl','wb'))
print('模型保存完成')
