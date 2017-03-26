#encoding=utf-8
import  csv
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import cross_validation

#得到评价指标rmspe_xg训练模型
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w
def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe
#该评价指标用来评价模型好坏
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe
#提出特征
def new_build_features_train(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)#对整个dataframe进行空白填充
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.append('StateHolidayA')
    features.append('StateHolidayB')
    features.append('StateHolidayC')

    data['StateHolidayA'] = 0
    data['StateHolidayB'] = 0
    data['StateHolidayC'] = 0

    data.loc[data['StateHoliday'] == 'a', 'StateHolidayA'] = '1'  #df.loc[筛选条件,列名]=value
    data.loc[data['StateHoliday'] == 'b', 'StateHolidayB'] = '1'
    data.loc[data['StateHoliday'] == 'c', 'StateHolidayC'] = '1'

    data['StateHolidayA'] = data['StateHolidayA'].astype(float)
    data['StateHolidayB'] = data['StateHolidayB'].astype(float)
    data['StateHolidayC'] = data['StateHolidayC'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')

    data['year'] = data.Date.apply(lambda x: x.split('-')[2])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[0])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[1])
    data['day'] = data['day'].astype(float)

    features.append('StoreTypeA')
    features.append('StoreTypeB')
    features.append('StoreTypeC')

    data['StoreTypeA'] = 0
    data['StoreTypeB'] = 0
    data['StoreTypeC'] = 0

    data.loc[data['StoreType'] == 'a', 'StoreTypeA'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreTypeB'] = '1'
    data.loc[data['StoreType'] == 'c', 'StoreTypeC'] = '1'

    data['StoreTypeA'] = data['StoreTypeA'].astype(float)
    data['StoreTypeB'] = data['StoreTypeB'].astype(float)
    data['StoreTypeC'] = data['StoreTypeC'].astype(float)

    features.append('AssortmentA')
    features.append('AssortmentB')

    data['AssortmentA'] = 0
    data['AssortmentB'] = 0

    data.loc[data['Assortment'] == 'a', 'AssortmentA'] = '1'
    data.loc[data['Assortment'] == 'b', 'AssortmentB'] = '1'

    data['AssortmentA'] = data['AssortmentA'].astype(float)
    data['AssortmentB'] = data['AssortmentB'].astype(float)
def get_submmit_data():
    pass

#设置参数
num_trees=450  #是怎么设置数量的？通过观察error大小？--------------------------------------------------------------------------------------
params = {"objective": "reg:linear",
          "eta": 0.15,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

DATA_DIR = "/home/wangtuntun/IJCAI/Rossman/Data/"
train_org = pd.read_csv(DATA_DIR + "train.csv")#返回dataframe
train_org = train_org[train_org["Open"] != 0]#dataframe直接筛选
store = pd.read_csv(DATA_DIR + "store.csv")
train_org = pd.merge(train_org, store, on='Store')#将两张表，按照store_id进行关联

#提取特征
features = []
new_build_features_train(features, train_org)

train,test=cross_validation.train_test_split(train_org,test_size=0.2)
X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)#将训练集划分为X_train和X_test用来训练模型
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))#features,target  加1的原因是担心目标为0
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))#比较好奇的是，为什么对target取对数。评价指标就这样？
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

#将数据和参数代入模型进行训练
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg,verbose_eval=True)

#将测试集代入模型进行预测
print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
predict_sales=np.exp(test_probs) - 1
# result_df=pd.DataFrame({"Store":test["Store"],"Real_target":test["Sales"],"Predict_target":predict_sales})
result_df=pd.DataFrame({"3":predict_sales,"2":test["Sales"],"1":test["Date"],"0":test["Store"]})
result_df_sorted=result_df.sort_values(by=["0","1"])
result_df_sorted.to_csv("/home/wangtuntun/result_3.csv",index=False)

#计算误差
indices = test_probs < 0
test_probs[indices] = 0
# error = rmspe(np.exp(test_probs) - 1, test['Sales'].values)
error = rmspe(predict_sales, test['Sales'])
print('error', error)#('error', 0.11964850182868811)

#预测未来六周的流量
future_data=pd.read_csv(DATA_DIR + "test.csv")#返回dataframe
future_data.drop("Id",axis=1,inplace=True)#test.csv文件多了"id"属性，少了"sales" "customer"属性
future_data["Sales"]=0
future_data["Customers"]=0
# future_data=future_data[future_data["Open"] != 0]#预测未来数据不能这样搞，会缺失某一天的预测。
#此外，给出的预测数据并不是完全 store_id  第一天，最后一天       可能某天缺少某个store_id,最后导致提交的数据没有某个store的信息。
#所以还是要自己生成预测数据
future_data=pd.merge(future_data, store, on='Store')
new_build_features_train([], future_data)
future_predict_pro=gbm.predict(xgb.DMatrix(future_data[features]))
future_sales=np.exp(future_predict_pro) - 1
future_df=pd.DataFrame({"2":future_sales,"1":future_data["Date"],"0":future_data["Store"]})
future_predict_sorted=future_df.sort_values(by=["0","1"])
future_predict_sorted.to_csv("/home/wangtuntun/result_4.csv")