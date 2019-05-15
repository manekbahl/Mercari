import pandas as pd  
import numpy as np    
import urllib        
import re            
import datetime      
import calendar      
import time          
import scipy         
#from sklearn.cluster import KMeans
import math          
import seaborn as sns 
import matplotlib.pyplot as plt
import os               
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
#from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import Ridge as Ridge
def calculate_RMSLE(actual,predicted):
    n= len(predicted)
    score = pow(sum(pow(np.log(np.array(predicted) + 1)-np.log(np.array(actual) + 1),2))/n,0.5)
    return score

def split_into_categories(item):
    try:
        text = item
        part1, part2, part3 = text.split('/')
        return part1, part2, part3
    except:
        return np.nan, np.nan, np.nan

data= pd.read_csv('train.tsv', sep='\t')
price_col = data['price'].values
data=data.drop('price', axis=1)
tr, tst, price_tr, price_tst = train_test_split(data.values, price_col, test_size=0.3, random_state=1992)
tr=pd.DataFrame(tr)
test_df=pd.DataFrame(tst)
price_tr=pd.DataFrame(price_tr)
actual_test_price=pd.Series(price_tst)
train_df = pd.concat([tr.reset_index(drop=True),price_tr], axis=1)
train_df.columns=list(data.columns) + ['price']
test_df.columns=list(data.columns)
test_id=pd.Series(test_df['train_id'].values)

train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: split_into_categories(val)))
test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: split_into_categories(val)))
train_df.head()

test_df.head()

cat_key1 = train_df.cat_1.unique().tolist() + test_df.cat_1.unique().tolist()
cat_key1 = list(set(cat_key1))
values = list(range(cat_key1.__len__()))
cat1_dict = dict(zip(cat_key1, values))

cat_key2 = train_df.cat_2.unique().tolist() + test_df.cat_2.unique().tolist()
cat_key2 = list(set(cat_key2))
values2 = list(range(cat_key2.__len__()))
cat2_dict = dict(zip(cat_key2, values2))

cat_key3 = train_df.cat_3.unique().tolist() + test_df.cat_3.unique().tolist()
cat_key3 = list(set(cat_key3))
values3 = list(range(cat_key3.__len__()))
cat3_dict = dict(zip(cat_key3, values3))

# Create the category labels
def cat_lab(item,cat1_dict = cat1_dict, cat2_dict = cat2_dict, cat3_dict = cat3_dict):
    part1 = item['cat_1']
    part2 = item['cat_2']
    part3 = item['cat_3']
    return cat1_dict[part1], cat2_dict[part2], cat3_dict[part3]

train_df["cat_1_label"], train_df["cat_2_label"], train_df["cat_3_label"] = zip(*train_df.apply(lambda val: cat_lab(val), axis =1))
test_df["cat_1_label"], test_df["cat_2_label"], test_df["cat_3_label"] = zip(*test_df.apply(lambda val: cat_lab(val), axis =1))
train_df.head(10)

#function create a column of binary column indicating the presence of Category name
def if_present(item):
    if item == item:
        return 1
    else:
        return 0
    
train_df['if_cat'] = train_df.category_name.apply(lambda item : if_present(item))
test_df['if_cat'] = test_df.category_name.apply(lambda item : if_present(item))
train_df.head()

train_df['if_brand'] = train_df.brand_name.apply(lambda item : if_present(item))
test_df['if_brand'] = test_df.brand_name.apply(lambda item : if_present(item))
train_df.head()

keys = train_df.brand_name.dropna().unique()
values = list(range(keys.__len__()))
brand_dict = dict(zip(keys, values))

#function to assign brand label
def brand_label(item):
    try:
        return brand_dict[item]
    except:
        return np.nan

train_df['brand_label'] = train_df.brand_name.apply(lambda item: brand_label(item))
test_df['brand_label'] = test_df.brand_name.apply(lambda item: brand_label(item))
train_df.head()

#function create a column of binary column indicating the presence of Item description
def if_description(item):
    if item == 'No description yet':
        a = 0
    else:
        a = 1
    return a

train_df['if_description'] = train_df.item_description.apply(lambda item : if_description(item))
test_df['if_description'] = test_df.item_description.apply(lambda item : if_description(item))
train_df.head()

#Remove Nulls in item description in train or test as tf-idf is not defined on nan

train_df = train_df.loc[train_df.item_description == train_df.item_description]
test_df = test_df.loc[test_df.item_description == test_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
test_df = test_df.loc[test_df.name == test_df.name]

#Perform TF-IDF on the item-description column

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['item_description'].values.tolist() + test_df['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['item_description'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)


#Replace the NAs with 0
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

len(data.name.unique())
train = train_df.copy()
test = test_df.copy()

# delete the backup dataframes to free up the memory
del train_df
del test_df
del train_svd
del test_svd

#Exclude the old features from which new features have been created.
columns_to_exclude = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
columns_to_include = [f for f in train.columns if f not in columns_to_exclude]

# Take log of the training prices since we need to minimize RMSLE and not RMSE
y = np.log(train['price'].values + 1)
del train['price']


##################################### Running Random forest 

x_train,y_train = train[columns_to_include],y

# Adjust the parameters by trial and error
model_rf = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)

#Fit the model
model_rf.fit(x_train, y_train)
model_rf.score(x_train,y_train)

#Predict on the test data
predicted_price_rf = model_rf.predict(test[columns_to_include])

#Take antilog of the prediction to get the predicted prices
predicted_price_rf = pd.Series(np.exp(predicted_price_rf)-1)

submit_rf = pd.concat([test_id,pd.Series(predicted_price_rf)],axis=1)
submit_rf.columns = ['test_id','price']
submit_rf.to_csv("TF_IDF_RF.csv", index=False)
RMSLE_rf = calculate_RMSLE(actual_test_price,predicted_price_rf)
print(RMSLE_rf)

##################################### Running XGboost 

Xtr, Xv, ytr, yv = train_test_split(train[columns_to_include].values, y, test_size=0.2, random_state=1992)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[columns_to_include].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model_xgb = xgb.train(xgb_par, dtrain, 80, watchlist, early_stopping_rounds=20, maximize=False, verbose_eval=20)

yvalid = model_xgb.predict(dvalid)
ytest = model_xgb.predict(dtest)


predicted_price_xgb = np.exp(ytest) - 1
submit_xgb = pd.concat([test_id,pd.Series(predicted_price_xgb)],axis=1)
submit_xgb.columns = ['test_id','price']
submit_xgb.to_csv("TF_ITF_XGB.csv", index=False)
RMSLE_xgb = calculate_RMSLE(actual_test_price,predicted_price_xgb)
print(RMSLE_xgb)

##################################### Running Ridge regression


x_train,y_train = train[columns_to_include],y
model_ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=1992)
model_ridge.fit(x_train, y_train)
predicted_price_ridge = model_ridge.predict(test[columns_to_include])
predicted_price_ridge = pd.Series(np.exp(predicted_price_ridge)-1)
submit_ridge = pd.concat([test_id,pd.Series(predicted_price_ridge)],axis=1)
submit_ridge.columns = ['test_id','price']
submit_ridge.to_csv("TF_IDF_RIDGE.csv", index=False)
RMSLE = calculate_RMSLE(actual_test_price,predicted_price_ridge)
print(RMSLE)

