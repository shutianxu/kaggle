# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import lightgbm as lgb


tap_fun_train = pd.read_csv("D:/data/tap4fun/tap_fun_train.csv", sep = ',')
tap_fun_test = pd.read_csv("D:/data/tap4fun/tap_fun_test.csv", sep = ',')
tap_fun_train['classification_label'] = tap_fun_train['prediction_pay_price'].map(lambda x: 1 if x >= 1 else 0)


'''
basic做数据预处理
'''
def data_process(X):
    X['wood'] = X['wood_add_value'] - X['wood_reduce_value'] 
    X['stone'] = X['stone_add_value'] - X['stone_reduce_value'] 
    X['ivory'] = X['ivory_add_value']  - X['ivory_reduce_value'] 
    X['meat'] =  X['meat_add_value'] - X['meat_reduce_value']
    X['magic'] =  X['magic_add_value'] - X['magic_reduce_value']
    X['infantry'] = X['infantry_add_value'] - X['infantry_reduce_value'] 
    X['cavalry'] = X['cavalry_add_value'] - X['cavalry_reduce_value'] 
    X['shaman'] = X['shaman_add_value'] - X['shaman_reduce_value'] 
    X['wound_infantry'] = X['wound_infantry_add_value'] - X['wound_infantry_reduce_value'] 
    X['wound_cavalry'] =  X['wound_cavalry_add_value'] - X['wound_cavalry_reduce_value']
    X['wound_shaman'] = X['wound_shaman_add_value'] - X['wound_shaman_reduce_value'] 
    X['general_acceleration'] = X['general_acceleration_reduce_value'] - X['general_acceleration_add_value'] 
    X['building_acceleration'] = X['building_acceleration_add_value'] - X['building_acceleration_reduce_value']
    X['reaserch_acceleration'] = X['reaserch_acceleration_add_value'] -  X['reaserch_acceleration_reduce_value']
    X['training_acceleration'] =  X['training_acceleration_add_value'] - X['training_acceleration_reduce_value']
    X['treatment_acceleration'] =  X['treatment_acceleraion_add_value'] - X['treatment_acceleration_reduce_value']  
# =============================================================================
#     X = X.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)
# =============================================================================
    return X

def data_clf_process(X):
    
# =============================================================================
#     X['bd_training_hut_level'] = X['bd_training_hut_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_healing_lodge_level'] = X['bd_healing_lodge_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_stronghold_level'] = X['bd_stronghold_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_outpost_portal_level'] = X['bd_outpost_portal_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_barrack_level'] = X['bd_barrack_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_healing_spring_level'] = X['bd_healing_spring_level'].map(lambda x: 1 if x > 0 else 0)
#     X['bd_dolmen_level'] = X['bd_dolmen_level'].map(lambda x: 1 if x > 0 else 0)
#  
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
    
    X = X
    return X



def data_reg_process(X):
    X = X
    return X


def get_dummies(X):
    X = pd.get_dummies(X, columns = ['bd_training_hut_level' ,'bd_healing_lodge_level' ,'bd_stronghold_level' ,'bd_outpost_portal_level' ,'bd_barrack_level' ,'bd_healing_spring_level' ,'bd_dolmen_level' ,'bd_guest_cavern_level' ,'bd_warehouse_level' ,'bd_watchtower_level' ,'bd_magic_coin_tree_level' ,'bd_hall_of_war_level' ,'bd_market_level' ,'bd_hero_gacha_level' ,'bd_hero_strengthen_level' ,'bd_hero_pve_level' ])
    
    return X


def min_max_scaler(X):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features_new = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(features_new, columns=X.columns)
    return X



data = data_process(tap_fun_train)



'''
'''



'''
回归建模测试集数据准备
'''


reg_test = tap_fun_test.drop(['user_id','register_time'],axis=1)
reg_test = data_process(reg_test)



'''
训练回归模型
'''
from lightgbm import LGBMRegressor


reg_target = data['classification_label']
reg_features = data.drop(['user_id','register_time','classification_label','prediction_pay_price'],axis=1)
reg_features = data_clf_process(reg_features)


X_train,X_test,y_train,y_test = train_test_split(
reg_features,reg_target,test_size=0.25,random_state=42)

'''
'''

reg_features = data_reg_process(reg_features)



reg = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=1000,min_child_weight=10,subsample=0.7, 
                    colsample_bytree=0.7,reg_alpha=0,learning_rate=0.1,reg_lambda=0.5,bagging_fraction = 0.8,
                    bagging_freq = 5,feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9)




'''
实际回归建模
'''

reg.fit(reg_features, reg_target, eval_set=[(reg_features, reg_target)], eval_metric='rmse',early_stopping_rounds=100)
reg_y_pre = reg.predict(reg_test)

y_pred = reg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


reg_df = pd.DataFrame({ 
                        'user_id' : tap_fun_test['user_id'],
                        'prediction_pay_price' : reg_y_pre
                        })


reg_df.to_csv('D:/999github/kaggle/sub_sample.csv', index=False)


'''
回归特征权重显示
'''

importances = reg.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(reg_features.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features.columns[indices[f]] ))
