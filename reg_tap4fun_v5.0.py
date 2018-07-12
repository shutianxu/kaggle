# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
"""
'''
V1.0回归  ->85
v2.0 LASSO回归 转化label 为 ([prediction_pay_price]-[pay_price])/[prediction_pay_price] -> 66
v3.0 LASSO回归 转化label 为 ([prediction_pay_price]-[pay_price])/[prediction_pay_price] -> 66

'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import datetime




# =============================================================================
# import sys
# sys.path.append('..')
# import gluonbook as gb
# from mxnet import autograd, gluon, init, nd
# from mxnet.gluon import loss as gloss, nn
# =============================================================================



'''
basic做数据预处理
'''
def data_process(X):
# =============================================================================
#     X['wood'] = X['wood_add_value'] - X['wood_reduce_value'] 
#     X['stone'] = X['stone_add_value'] - X['stone_reduce_value'] 
#     X['ivory'] = X['ivory_add_value']  - X['ivory_reduce_value'] 
#     X['meat'] =  X['meat_add_value'] - X['meat_reduce_value']
#     X['magic'] =  X['magic_add_value'] - X['magic_reduce_value']
#     X['infantry'] = X['infantry_add_value'] - X['infantry_reduce_value'] 
#     X['cavalry'] = X['cavalry_add_value'] - X['cavalry_reduce_value'] 
#     X['shaman'] = X['shaman_add_value'] - X['shaman_reduce_value'] 
#     X['wound_infantry'] = X['wound_infantry_add_value'] - X['wound_infantry_reduce_value'] 
#     X['wound_cavalry'] =  X['wound_cavalry_add_value'] - X['wound_cavalry_reduce_value']
#     X['wound_shaman'] = X['wound_shaman_add_value'] - X['wound_shaman_reduce_value'] 
#     X['general_acceleration'] = X['general_acceleration_reduce_value'] - X['general_acceleration_add_value'] 
#     X['building_acceleration'] = X['building_acceleration_add_value'] - X['building_acceleration_reduce_value']
#     X['reaserch_acceleration'] = X['reaserch_acceleration_add_value'] -  X['reaserch_acceleration_reduce_value']
#     X['training_acceleration'] =  X['training_acceleration_add_value'] - X['training_acceleration_reduce_value']
#     X['treatment_acceleration'] =  X['treatment_acceleraion_add_value'] - X['treatment_acceleration_reduce_value']  
# =============================================================================
    X = X
# =============================================================================
#     X = X.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)
# =============================================================================
    return X


def get_dummies(X):
    X = pd.get_dummies(X, columns = ['bd_training_hut_level' ,'bd_healing_lodge_level' ,'bd_stronghold_level' ,'bd_outpost_portal_level' ,'bd_barrack_level' ,'bd_healing_spring_level' ,'bd_dolmen_level' ,'bd_guest_cavern_level' ,'bd_warehouse_level' ,'bd_watchtower_level' ,'bd_magic_coin_tree_level' ,'bd_hall_of_war_level' ,'bd_market_level' ,'bd_hero_gacha_level' ,'bd_hero_strengthen_level' ,'bd_hero_pve_level','sr_scout_level' ,'sr_training_speed_level' ,'sr_infantry_tier_2_level' ,'sr_cavalry_tier_2_level' ,'sr_shaman_tier_2_level' ,'sr_infantry_atk_level' ,'sr_cavalry_atk_level' ,'sr_shaman_atk_level' ,'sr_infantry_tier_3_level' ,'sr_cavalry_tier_3_level' ,'sr_shaman_tier_3_level' ,'sr_troop_defense_level' ,'sr_infantry_def_level' ,'sr_cavalry_def_level' ,'sr_shaman_def_level' ,'sr_infantry_hp_level' ,'sr_cavalry_hp_level' ,'sr_shaman_hp_level' ,'sr_infantry_tier_4_level' ,'sr_cavalry_tier_4_level' ,'sr_shaman_tier_4_level' ,'sr_troop_attack_level' ,'sr_construction_speed_level' ,'sr_hide_storage_level' ,'sr_troop_consumption_level' ,'sr_rss_b_prod_level' ,'sr_rss_c_prod_level' ,'sr_rss_d_prod_level' ,'sr_rss_a_gather_level' ,'sr_rss_b_gather_level' ,'sr_rss_c_gather_level' ,'sr_rss_d_gather_level' ,'sr_troop_load_level' ,'sr_rss_e_gather_level' ,'sr_rss_e_prod_level' ,'sr_outpost_durability_level' ,'sr_outpost_tier_2_level' ,'sr_healing_space_level' ,'sr_gathering_hunter_buff_level' ,'sr_healing_speed_level' ,'sr_outpost_tier_3_level' ,'sr_alliance_march_speed_level' ,'sr_pvp_march_speed_level' ,'sr_gathering_march_speed_level' ,'sr_outpost_tier_4_level' ,'sr_guest_troop_capacity_level' ,'sr_march_size_level' ,'sr_rss_help_bonus_level' ])   
# =============================================================================
#     X = pd.get_dummies(X, columns = ['sr_rss_e_prod_level','sr_gathering_march_speed_level','sr_outpost_durability_level','bd_hall_of_war_level','sr_rss_e_gather_level','sr_healing_space_level','sr_rss_a_gather_level','sr_hide_storage_level'])
# =============================================================================
    return X

def standardscaler(X):    
    standardscaler = preprocessing.StandardScaler()
    features_new = standardscaler.fit_transform(X)
    X = pd.DataFrame(features_new, columns=X.columns)
    return X

tap_fun_train = pd.read_csv("D:/data/tap4fun/tap_fun_train.csv", sep = ',')
tap_fun_test = pd.read_csv("D:/data/tap4fun/tap_fun_test.csv", sep = ',')




'''
数据可视化分析
'''
# =============================================================================
# data_vis = tap_fun_train[tap_fun_train.prediction_pay_price > 0]
# corrmat = data_vis.corr()
# f, ax = plt.subplots(figsize=(40, 40))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# 
# 
# 
# k  = 20 # 关系矩阵中将显示10个特征
# cols = corrmat.nlargest(k, 'prediction_pay_price')['prediction_pay_price'].index
# cm = np.corrcoef(data_vis[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, \
#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# 
# 
# =============================================================================



tap_fun_train['xiangcha'] = tap_fun_train['prediction_pay_price']-tap_fun_train['pay_price']
tap_fun_train['label'] = tap_fun_train['prediction_pay_price']
tap_fun_train = tap_fun_train.fillna(0)

tap_fun_train['classification_label'] = tap_fun_train['xiangcha'].map(lambda x: 1 if x > 0 else 0)
data = pd.concat([tap_fun_train,tap_fun_test])
data = data.fillna(-1)
register_time = []
for i in data['register_time']:
# =============================================================================
#     print(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime("%a"))
# =============================================================================
    register_time.append(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime("%a"))
data['register_time'] = register_time

print(data['classification_label'].value_counts())

'''
分类数据预处理
'''
data = data_process(data)
# =============================================================================
# data = get_dummies(data)
# =============================================================================
print('dummies finish!!')


'''
回归建模测试集数据准备
'''

# =============================================================================
# reg_data = get_dummies(reg_data)
# =============================================================================
reg_test_1 = data[(data.classification_label == -1)&(data.pay_price == 0)].drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
reg_test_1['intercept'] = 1.0
# =============================================================================
# reg_test_2 = data[(data.classification_label == -1)&(data.pay_price > 0)].drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
# =============================================================================

reg_test_2 = data[(data.classification_label == -1)&(data.pay_price > 0)&(data.pay_price < 100)][['pay_price','ivory_add_value','stone_add_value','ivory_reduce_value','wood_add_value','general_acceleration_add_value','stone_reduce_value','training_acceleration_add_value','wood_reduce_value','meat_add_value','general_acceleration_reduce_value']]
reg_test_2['intercept'] = 1.0


reg_test_3 = data[(data.classification_label == -1)&(data.pay_price >= 100)][['pay_price','ivory_add_value','stone_add_value','ivory_reduce_value','wood_add_value','general_acceleration_add_value','stone_reduce_value','training_acceleration_add_value','wood_reduce_value','meat_add_value','general_acceleration_reduce_value']]
reg_test_3['intercept'] = 1.0
'''
训练回归模型
'''
reg_data = data
reg_data_1 = reg_data[(reg_data.label >= 0)&(reg_data.pay_price == 0)]
# =============================================================================
# reg_data = reg_data.drop((reg_data[(reg_data.pay_price >= 100) & (reg_data.avg_online_minutes < 35)]).index)
# reg_data_1 = reg_data_1.drop((reg_data_1[(reg_data_1.pay_price < 0.99) & (reg_data_1.avg_online_minutes > 840)]).index)
# =============================================================================
reg_target_1 = reg_data_1['label']
reg_features_1 = reg_data_1.drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
reg_features_1['intercept'] = 1.0



reg_data_2 = reg_data[(reg_data.label >= 0)&(reg_data.pay_price > 0)&(reg_data.pay_price < 100)]
reg_target_2 = reg_data_2['label']
# =============================================================================
# reg_features_2 = reg_data_2.drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
# =============================================================================
reg_features_2 = reg_data_2[['pay_price','ivory_add_value','stone_add_value','ivory_reduce_value','wood_add_value','general_acceleration_add_value','stone_reduce_value','training_acceleration_add_value','wood_reduce_value','meat_add_value','general_acceleration_reduce_value']]
reg_features_2['intercept'] = 1.0





reg_data_3 = reg_data[(reg_data.label >= 0)&(reg_data.pay_price >= 100)]
# =============================================================================
# reg_data_3 = reg_data_3.drop((reg_data_3[(reg_data_3.pay_price >= 1000) & (reg_data_3.pay_price == reg_data_3.prediction_pay_price)]).index)
# =============================================================================
reg_target_3 = reg_data_3['label']
# =============================================================================
# reg_features_3 = reg_data_3.drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
# =============================================================================
reg_features_3 = reg_data_3[['pay_price','ivory_add_value','stone_add_value','ivory_reduce_value','wood_add_value','general_acceleration_add_value','stone_reduce_value','training_acceleration_add_value','wood_reduce_value','meat_add_value','general_acceleration_reduce_value']]
reg_features_3['intercept'] = 1.0



'''
'''
# =============================================================================
# from sklearn import linear_model
# reg = linear_model.Lasso(alpha=0.1, fit_intercept=True, normalize=True, precompute=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=True, positive=True, random_state=42, selection='cyclic')
# =============================================================================

from sklearn.linear_model import ElasticNet
reg_1 = ElasticNet(alpha=0.8, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True, max_iter=1000, tol=0.001, warm_start=True, positive=True, random_state=42, selection='cyclic')
reg_2 = ElasticNet(alpha=0.8, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True, max_iter=1000, tol=0.001, warm_start=True, positive=True, random_state=42, selection='cyclic')
reg_3 = ElasticNet(alpha=0.8, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True, max_iter=1000, tol=0.001, warm_start=True, positive=True, random_state=42, selection='cyclic')

'''
实际回归建模
'''


X_train,X_test,y_train,y_test = train_test_split(reg_features_1,reg_target_1,test_size=0.2,random_state=42)
reg_1.fit(X_train, y_train)
y_pre = reg_1.predict(X_test)


print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
y_pred_1 = reg_1.predict(reg_test_1)
    

reg_df_1 = pd.DataFrame({ 
                        'user_id' : data[(data.classification_label == -1)&(data.pay_price == 0)]['user_id'],
                        'price':reg_test_1['pay_price'],
                        'prediction_pay_price' : y_pred_1
                        })
 
reg_df_1['prediction_pay_price'] = reg_df_1['prediction_pay_price'].map(lambda x: 0 if x < 0 else x)
    
X_train,X_test,y_train,y_test = train_test_split(reg_features_2,reg_target_2,test_size=0.2,random_state=42)
reg_2.fit(X_train, y_train)
y_pre = reg_2.predict(X_test)


print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
y_pred_2 = reg_2.predict(reg_test_2)
    

reg_df_2 = pd.DataFrame({ 
                        'user_id' : data[(data.classification_label == -1)&(data.pay_price > 0)&(data.pay_price <100)]['user_id'],
                        'price':reg_test_2['pay_price'],
                        'prediction_pay_price' : y_pred_2
                        })
reg_df_2['prediction_pay_price'] = reg_df_2['prediction_pay_price'].map(lambda x: 0.99 if x < 0 else x)




X_train,X_test,y_train,y_test = train_test_split(reg_features_3,reg_target_3,test_size=0.2,random_state=42)
reg_3.fit(X_train, y_train)
y_pre = reg_3.predict(X_test)


print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
y_pred_3 = reg_3.predict(reg_test_3)
    

reg_df_3 = pd.DataFrame({ 
                        'user_id' : data[(data.classification_label == -1)&(data.pay_price >= 100)]['user_id'],
                        'price':reg_test_3['pay_price'],
                        'prediction_pay_price' : y_pred_3
                        })
reg_df_3['prediction_pay_price'] = reg_df_3['prediction_pay_price'].map(lambda x: 0.99 if x < 0 else x)










reg_df = pd.concat([reg_df_1,reg_df_2,reg_df_3])

reg_df['prediction_pay_price'] = reg_df['prediction_pay_price'].map(lambda x: 0 if x < 0.99 else x)

reg_df['a'] = reg_df['prediction_pay_price'] - reg_df['price']
reg_df['a'] = reg_df['a'].map(lambda x: x if x < 0 else 0)
reg_df['prediction_pay_price'] = reg_df['prediction_pay_price'] - reg_df['a']


reg_df[['user_id','prediction_pay_price']].to_csv('D:/999github/kaggle/sub_sample.csv', index=False)
reg_df[['user_id','price','prediction_pay_price']].to_csv('D:/999github/kaggle/sub_sample_10.csv', index=False)


'''
回归特征权重显示
'''
importances = reg_1.coef_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(reg_features_1.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features_1.columns[indices[f]] ))



importances = reg_2.coef_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(reg_features_2.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features_2.columns[indices[f]] ))


importances = reg_3.coef_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(reg_features_3.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features_3.columns[indices[f]] ))
