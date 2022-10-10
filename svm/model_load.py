import pickle
import pandas as pd
pd.set_option('display.max_rows', None, 'display.max_columns', None)
#导入数据
data=pd.read_csv(r'G:\kaggle_datas\UCI_Credit_Card\UCI_Credit_Card.csv')
#Load X Variables into a Pandas Dataframe with columns
X = data.drop(['ID','default.payment.next.month'], axis = 1)
model = pickle.load(open('svc_model.pkl', 'rb'))



print(model.predict(X))

# [1 1 0 ... 1 1 1]








