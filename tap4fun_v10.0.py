# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
"""
'''
V1.0回归  ->85
v2.0回归+分类，特征选取 -> 82
V3.0数据集+测试集合并，回归模型做哑变量处理 -> 78.399
V4.0特征工程,分类切片哑变量处理，加入注册时间特征 ->  77.3297
V5.0
加入回归算法PCA降维 200维-> 94（待优化）
加入回归算法PCA降维 500维-> 98（待优化）
V6.0
调整分类lambda(0切),封箱在线时长、pay_prace -> 84.47
调整分类lambda(1切),封箱在线时长、pay_prace -> 84.31
V7.0特征处理完直接回归 -> 0
V8.0 调整分类lambda(1切，>0.5，reg_alpha=1) -> 84
V9.0 加入验证集,回归 -> 81
V10.0 45-7 相差作为LABEL
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import lightgbm as lgb
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import datetime



'''
basic做数据预处理
'''
def data_process(X):
    X = X
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
# =============================================================================
#     X = X.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)
# =============================================================================
    return X

def data_clf_process(X):
    X = X
    X = X.drop(X[(X.avg_online_minutes < 5)&(X.prediction_pay_price > 1000)].index)
    X = X.drop(X[(X.avg_online_minutes >1000)&(X.prediction_pay_price < 10)].index)  
# =============================================================================
#     X['bd_training_hut_level'] = X['bd_training_hut_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_healing_lodge_level'] = X['bd_healing_lodge_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_stronghold_level'] = X['bd_stronghold_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_outpost_portal_level'] = X['bd_outpost_portal_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_barrack_level'] = X['bd_barrack_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_healing_spring_level'] = X['bd_healing_spring_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_dolmen_level'] = X['bd_dolmen_level'].map(lambda x: 1 if x > 0 else 0) 
#     X['bd_guest_cavern_level'] = X['bd_guest_cavern_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_warehouse_level'] = X['bd_warehouse_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_watchtower_level'] = X['bd_watchtower_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_magic_coin_tree_level'] = X['bd_magic_coin_tree_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_hall_of_war_level'] = X['bd_hall_of_war_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_market_level'] = X['bd_market_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_hero_gacha_level'] = X['bd_hero_gacha_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_hero_strengthen_level'] = X['bd_hero_strengthen_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_hero_pve_level'] = X['bd_hero_pve_level'].map(lambda x: 1 if x > 0 else 0)
# =============================================================================
    return X

def data_reg_process(X):
    X = X
    X = X.drop(X[(X.avg_online_minutes < 5)&(X.prediction_pay_price > 1000)].index)
    X = X.drop(X[(X.avg_online_minutes >1000)&(X.prediction_pay_price < 10)].index)
    return X

def get_dummies(X):
    X = pd.get_dummies(X, columns = ['register_time','bd_training_hut_level' ,'bd_healing_lodge_level' ,'bd_stronghold_level' ,'bd_outpost_portal_level' ,'bd_barrack_level' ,'bd_healing_spring_level' ,'bd_dolmen_level' ,'bd_guest_cavern_level' ,'bd_warehouse_level' ,'bd_watchtower_level' ,'bd_magic_coin_tree_level' ,'bd_hall_of_war_level' ,'bd_market_level' ,'bd_hero_gacha_level' ,'bd_hero_strengthen_level' ,'bd_hero_pve_level','sr_scout_level' ,'sr_training_speed_level' ,'sr_infantry_tier_2_level' ,'sr_cavalry_tier_2_level' ,'sr_shaman_tier_2_level' ,'sr_infantry_atk_level' ,'sr_cavalry_atk_level' ,'sr_shaman_atk_level' ,'sr_infantry_tier_3_level' ,'sr_cavalry_tier_3_level' ,'sr_shaman_tier_3_level' ,'sr_troop_defense_level' ,'sr_infantry_def_level' ,'sr_cavalry_def_level' ,'sr_shaman_def_level' ,'sr_infantry_hp_level' ,'sr_cavalry_hp_level' ,'sr_shaman_hp_level' ,'sr_infantry_tier_4_level' ,'sr_cavalry_tier_4_level' ,'sr_shaman_tier_4_level' ,'sr_troop_attack_level' ,'sr_construction_speed_level' ,'sr_hide_storage_level' ,'sr_troop_consumption_level' ,'sr_rss_b_prod_level' ,'sr_rss_c_prod_level' ,'sr_rss_d_prod_level' ,'sr_rss_a_gather_level' ,'sr_rss_b_gather_level' ,'sr_rss_c_gather_level' ,'sr_rss_d_gather_level' ,'sr_troop_load_level' ,'sr_rss_e_gather_level' ,'sr_rss_e_prod_level' ,'sr_outpost_durability_level' ,'sr_outpost_tier_2_level' ,'sr_healing_space_level' ,'sr_gathering_hunter_buff_level' ,'sr_healing_speed_level' ,'sr_outpost_tier_3_level' ,'sr_alliance_march_speed_level' ,'sr_pvp_march_speed_level' ,'sr_gathering_march_speed_level' ,'sr_outpost_tier_4_level' ,'sr_guest_troop_capacity_level' ,'sr_march_size_level' ,'sr_rss_help_bonus_level' ])   
# =============================================================================
#     X = pd.get_dummies(X, columns = ['register_time'])   
# =============================================================================
    return X

def min_max_scaler(X):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features_new = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(features_new, columns=X.columns)
    return X

def pca_process(X):
    pca = PCA(n_components= 500)
    X = pca.fit_transform(X)
    return X

tap_fun_train = pd.read_csv("D:/data/tap4fun/tap_fun_train.csv", sep = ',')
tap_fun_test = pd.read_csv("D:/data/tap4fun/tap_fun_test.csv", sep = ',',)




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


# =============================================================================
# online_labels = ['0', '60', '120', '180', '240', '300', '360', '420' , '480' , '540', '600','other']
# data['online_group'] = pd.cut(data.avg_online_minutes, [0,60,120,180,240,300,360,420,480,540,600,660,2040], right=True, labels=online_labels)
# 
# pay_price_labels = ['0' ,'5' ,'10' ,'15' ,'20' ,'25' ,'30' ,'35' ,'40' ,'45' ,'50' ,'55' ,'60' ,'65' ,'70' ,'75' ,'80' ,'85' ,'90' ,'95' ,'100','other']
# data['pay_price_group'] = pd.cut(data.pay_price, [0,5 ,10 ,15 ,20 ,25 ,30 ,35 ,40 ,45 ,50 ,55 ,60 ,65 ,70 ,75 ,80 ,85 ,90 ,95 ,100,105,7455], right=True, labels=pay_price_labels)
# =============================================================================



print(data['classification_label'].value_counts())

'''
分类数据预处理
'''
data = data_process(data)
data = get_dummies(data)
print('dummies finish!!')
'''
test
'''
clf_data = data[data.classification_label != -1]
clf_data = data_clf_process(clf_data)
target = clf_data['classification_label']
features = clf_data.drop(['user_id','classification_label','prediction_pay_price','xiangcha'],axis=1)


'''
分类调参
'''


clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=127, reg_alpha=0, reg_lambda=1,
        max_depth=8, n_estimators=800, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=2,
        learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=-1,class_weight = 'balanced'
    )

'''
实际分类建模
'''
clf_test = data[data.classification_label == -1]
clf_test = clf_test.drop(['user_id','classification_label','prediction_pay_price','xiangcha'],axis=1)



cnt=1
size = math.ceil(len(features) / cnt)
result=[]
print('ready for classification!!')
for i in range(cnt):    
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(features) else len(features)
    slice_features = features[start:end]
    slice_target = target[start:end]
    X_train,X_test,y_train,y_test = train_test_split(slice_features,slice_target,test_size=0.3,random_state=42)
    clf.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test, y_test)],eval_names =['train_data','valid_data'], eval_metric='error',early_stopping_rounds=100) 
    y_pre = clf.predict(X_test)
    y_pre_prod = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test,y_pre))
    print(metrics.roc_auc_score(y_test,y_pre_prod))
    y_pre_pro = clf.predict_proba(clf_test)[:, 1]
    result.append(y_pre_pro)
y_pre_pro = np.mean(result,axis=0)
# =============================================================================
# y_pre = clf.predict(clf_test)
# =============================================================================



clf_df_final = pd.DataFrame({ 'user_id' : data[data.classification_label == -1]['user_id'],
                     'y_pre_pro' : y_pre_pro,
                        })
clf_df_final['y_pre'] = clf_df_final['y_pre_pro'].map(lambda x: 1 if x >= 0.5 else 0)
    
clf_df_final.to_csv('D:/999github/kaggle/clf_result.csv', index=False)



# =============================================================================
# y_pre = clf.predict(X_test)
# y_pre_pro = clf.predict_proba(X_test)[:, 1]
# print(y_pre_pro)
# print(classification_report(y_test,y_pre))
# print(metrics.roc_auc_score(y_test,y_pre_pro))
# =============================================================================

# =============================================================================
# '''
# 分类特征权重显示
# '''
# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]
# print("Feature ranking:")
# for f in range(features.shape[1]):
#     print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , features.columns[indices[f]] ))
# =============================================================================



# =============================================================================
# clf_df_final = pd.read_csv("D:/999github/kaggle/clf_result.csv", sep = ',')
# =============================================================================

'''
回归建模测试集数据准备
'''
reg_data = data_process(data)
# =============================================================================
# reg_data = get_dummies(reg_data)
# =============================================================================


clf_df = clf_df_final[clf_df_final.y_pre == 1]
choose_test = pd.merge(clf_df, reg_data, on='user_id')

reg_test = choose_test.drop(['user_id','y_pre','y_pre_pro','classification_label','prediction_pay_price','xiangcha'],axis=1)



'''
训练回归模型
'''
from lightgbm import LGBMRegressor

reg_data = reg_data[reg_data.xiangcha > 0]
reg_data = data_reg_process(reg_data)
reg_target = reg_data['xiangcha']
reg_features = reg_data.drop(['user_id','classification_label','prediction_pay_price','xiangcha'],axis=1)


'''
'''



reg = LGBMRegressor(num_leaves=127,max_depth=7,n_estimators=500,min_child_weight=20,subsample=0.7, 
                    colsample_bytree=0.7,reg_alpha=1,learning_rate=0.05,reg_lambda=0,bagging_fraction = 0.7,
                    bagging_freq = 5,feature_fraction_seed=8, bagging_seed=8)




'''
实际回归建模
'''
X_train,X_test,y_train,y_test = train_test_split(reg_features,reg_target,test_size=0.2,random_state=42)
reg.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test, y_test)],eval_names =['train_data','valid_data'], eval_metric='rmse',early_stopping_rounds=100) 

reg_y_pre = reg.predict(reg_test)

y_pred = reg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


reg_df = pd.DataFrame({ 
                        'user_id' : choose_test['user_id'],
                        'xiangcha' : reg_y_pre,
                        })


final_df = pd.merge(choose_test[['user_id','pay_price']], reg_df, on='user_id',how='left')
final_df['prediction_pay_price'] = final_df['xiangcha'] + final_df['pay_price']
final_df = final_df[['user_id','prediction_pay_price']]
df = pd.merge(clf_df_final, final_df, on='user_id',how='left')
df = df.fillna(0)
df = df[['user_id','prediction_pay_price']]
df.to_csv('D:/999github/kaggle/sub_sample.csv', index=False)


'''
回归特征权重显示
'''
# =============================================================================
# 
# importances = reg.feature_importances_
# indices = np.argsort(importances)[::-1]
# print("Feature ranking:")
# for f in range(reg_features.shape[1]):
#     print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features.columns[indices[f]] ))
# =============================================================================

