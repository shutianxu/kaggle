# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500

"""




import pandas as pd
import numpy as np
import math


tap_fun_train = pd.read_csv("D:/data/tap4fun/tap_fun_train.csv", sep = ',')
tap_fun_test = pd.read_csv("D:/data/tap4fun/tap_fun_test.csv", sep = ',')
tap_fun_train['classification_label'] = tap_fun_train['prediction_pay_price'].map(lambda x: 1 if x >= 1 else 0)


# =============================================================================
# '''
# nan值查看
# '''
# null_counts = tap_fun_train.isnull().sum()
# print(null_counts)
# =============================================================================

'''
basic做数据预处理
'''
tap_fun_train['wood'] = tap_fun_train['wood_reduce_value'] - tap_fun_train['wood_add_value']
tap_fun_train['stone'] = tap_fun_train['stone_reduce_value'] - tap_fun_train['stone_add_value']
tap_fun_train['ivory'] = tap_fun_train['ivory_reduce_value'] - tap_fun_train['ivory_add_value']
tap_fun_train['meat'] = tap_fun_train['meat_reduce_value'] - tap_fun_train['meat_add_value']
tap_fun_train['magic'] = tap_fun_train['magic_reduce_value'] - tap_fun_train['magic_add_value']
tap_fun_train['infantry'] = tap_fun_train['infantry_reduce_value'] - tap_fun_train['infantry_add_value']
tap_fun_train['cavalry'] = tap_fun_train['cavalry_reduce_value'] - tap_fun_train['cavalry_add_value']
tap_fun_train['shaman'] = tap_fun_train['shaman_reduce_value'] - tap_fun_train['shaman_add_value']
tap_fun_train['wound_infantry'] = tap_fun_train['wound_infantry_reduce_value'] - tap_fun_train['wound_infantry_add_value']
tap_fun_train['wound_cavalry'] = tap_fun_train['wound_cavalry_reduce_value'] - tap_fun_train['wound_cavalry_add_value']
tap_fun_train['wound_shaman'] = tap_fun_train['wound_shaman_reduce_value'] - tap_fun_train['wound_shaman_add_value']
tap_fun_train['general_acceleration'] = tap_fun_train['general_acceleration_reduce_value'] - tap_fun_train['general_acceleration_add_value']
tap_fun_train['building_acceleration'] = tap_fun_train['building_acceleration_reduce_value'] - tap_fun_train['building_acceleration_add_value']
tap_fun_train['reaserch_acceleration'] = tap_fun_train['reaserch_acceleration_reduce_value'] - tap_fun_train['reaserch_acceleration_add_value']
tap_fun_train['training_acceleration'] = tap_fun_train['training_acceleration_reduce_value'] - tap_fun_train['training_acceleration_add_value']
tap_fun_train['treatment_acceleration'] = tap_fun_train['treatment_acceleration_reduce_value'] - tap_fun_train['treatment_acceleraion_add_value']

tap_fun_train = tap_fun_train.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)
data = tap_fun_train

'''
过滤无用的字段
'''
orig_columns = data.columns
drop_columns = []
for col in orig_columns:
    col_series = data[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
data = data.drop(drop_columns, axis=1)
print(drop_columns)

target = data['classification_label']
features = data.drop(['user_id','register_time','classification_label','prediction_pay_price'],axis=1)
# =============================================================================
# data = pd.get_dummies(feature, columns = ['bd_training_hut_level' ,'bd_healing_lodge_level' ,'bd_stronghold_level' ,'bd_outpost_portal_level' ,'bd_barrack_level' ,'bd_healing_spring_level' ,'bd_dolmen_level' ,'bd_guest_cavern_level' ,'bd_warehouse_level' ,'bd_watchtower_level' ,'bd_magic_coin_tree_level' ,'bd_hall_of_war_level' ,'bd_market_level' ,'bd_hero_gacha_level' ,'bd_hero_strengthen_level' ,'bd_hero_pve_level' ,'sr_scout_level' ,'sr_training_speed_level' ,'sr_infantry_tier_2_level' ,'sr_cavalry_tier_2_level' ,'sr_shaman_tier_2_level' ,'sr_infantry_atk_level' ,'sr_cavalry_atk_level' ,'sr_shaman_atk_level' ,'sr_infantry_tier_3_level' ,'sr_cavalry_tier_3_level' ,'sr_shaman_tier_3_level' ,'sr_troop_defense_level' ,'sr_infantry_def_level' ,'sr_cavalry_def_level' ,'sr_shaman_def_level' ,'sr_infantry_hp_level' ,'sr_cavalry_hp_level' ,'sr_shaman_hp_level' ,'sr_infantry_tier_4_level' ,'sr_cavalry_tier_4_level' ,'sr_shaman_tier_4_level' ,'sr_troop_attack_level' ,'sr_construction_speed_level' ,'sr_hide_storage_level' ,'sr_troop_consumption_level' ,'sr_rss_b_prod_level' ,'sr_rss_c_prod_level' ,'sr_rss_d_prod_level' ,'sr_rss_a_gather_level' ,'sr_rss_b_gather_level' ,'sr_rss_c_gather_level' ,'sr_rss_d_gather_level' ,'sr_troop_load_level' ,'sr_rss_e_gather_level' ,'sr_rss_e_prod_level' ,'sr_outpost_durability_level' ,'sr_outpost_tier_2_level' ,'sr_healing_space_level' ,'sr_gathering_hunter_buff_level' ,'sr_healing_speed_level' ,'sr_outpost_tier_3_level' ,'sr_alliance_march_speed_level' ,'sr_pvp_march_speed_level' ,'sr_gathering_march_speed_level' ,'sr_outpost_tier_4_level' ,'sr_guest_troop_capacity_level' ,'sr_march_size_level' ,'sr_rss_help_bonus_level'])
# =============================================================================

# =============================================================================
# ##基于树的方法不用做标准化、归一化处理
# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# features_new = min_max_scaler.fit_transform(features)
# features = pd.DataFrame(features_new, columns=features.columns)
# =============================================================================


from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report   
from sklearn import metrics
import lightgbm as lgb



'''
分类调参
'''
X_train,X_test,y_train,y_test = train_test_split(
features,target,test_size=0.25,random_state=42)

clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=40, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=600, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=2,
        learning_rate=0.3, min_child_weight=50, random_state=2018, n_jobs=-1,class_weight = 'balanced'
    )
clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc',early_stopping_rounds=100)
y_pre = clf.predict(X_test)
y_pre_pro = clf.predict_proba(X_test)[:, 1]
print(y_pre_pro)
print(classification_report(y_test,y_pre))
print(metrics.roc_auc_score(y_test,y_pre_pro))


'''
实际分类建模
'''

test = tap_fun_test.drop(['user_id','register_time'],axis=1)

test['wood'] = test['wood_reduce_value'] - test['wood_add_value']
test['stone'] = test['stone_reduce_value'] - test['stone_add_value']
test['ivory'] = test['ivory_reduce_value'] - test['ivory_add_value']
test['meat'] = test['meat_reduce_value'] - test['meat_add_value']
test['magic'] = test['magic_reduce_value'] - test['magic_add_value']
test['infantry'] = test['infantry_reduce_value'] - test['infantry_add_value']
test['cavalry'] = test['cavalry_reduce_value'] - test['cavalry_add_value']
test['shaman'] = test['shaman_reduce_value'] - test['shaman_add_value']
test['wound_infantry'] = test['wound_infantry_reduce_value'] - test['wound_infantry_add_value']
test['wound_cavalry'] = test['wound_cavalry_reduce_value'] - test['wound_cavalry_add_value']
test['wound_shaman'] = test['wound_shaman_reduce_value'] - test['wound_shaman_add_value']
test['general_acceleration'] = test['general_acceleration_reduce_value'] - test['general_acceleration_add_value']
test['building_acceleration'] = test['building_acceleration_reduce_value'] - test['building_acceleration_add_value']
test['reaserch_acceleration'] = test['reaserch_acceleration_reduce_value'] - test['reaserch_acceleration_add_value']
test['training_acceleration'] = test['training_acceleration_reduce_value'] - test['training_acceleration_add_value']
test['treatment_acceleration'] = test['treatment_acceleration_reduce_value'] - test['treatment_acceleraion_add_value']


test = test.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)


clf.fit(features, target, eval_set=[(features, target)], eval_metric='auc',early_stopping_rounds=100)

y_pre = clf.predict(test)
y_pre_pro = clf.predict_proba(test)[:, 1]


clf_df_final = pd.DataFrame({ 'user_id' : tap_fun_test['user_id'],
                     'y_pre' : y_pre,
                     'y_pre_pro' : y_pre_pro,
                        })

clf_df_final.to_csv('D:/999github/kaggle/clf_result.csv', index=False)

'''
回归建模测试集数据准备
'''

clf_df = clf_df_final[clf_df_final.y_pre == 1]   
choose_test = pd.merge(clf_df, tap_fun_test, on='user_id')

reg_test = choose_test.drop(['user_id','register_time','y_pre','y_pre_pro'],axis=1)

reg_test['wood'] = reg_test['wood_reduce_value'] - reg_test['wood_add_value']
reg_test['stone'] = reg_test['stone_reduce_value'] - reg_test['stone_add_value']
reg_test['ivory'] = reg_test['ivory_reduce_value'] - reg_test['ivory_add_value']
reg_test['meat'] = reg_test['meat_reduce_value'] - reg_test['meat_add_value']
reg_test['magic'] = reg_test['magic_reduce_value'] - reg_test['magic_add_value']
reg_test['infantry'] = reg_test['infantry_reduce_value'] - reg_test['infantry_add_value']
reg_test['cavalry'] = reg_test['cavalry_reduce_value'] - reg_test['cavalry_add_value']
reg_test['shaman'] = reg_test['shaman_reduce_value'] - reg_test['shaman_add_value']
reg_test['wound_infantry'] = reg_test['wound_infantry_reduce_value'] - reg_test['wound_infantry_add_value']
reg_test['wound_cavalry'] = reg_test['wound_cavalry_reduce_value'] - reg_test['wound_cavalry_add_value']
reg_test['wound_shaman'] = reg_test['wound_shaman_reduce_value'] - reg_test['wound_shaman_add_value']
reg_test['general_acceleration'] = reg_test['general_acceleration_reduce_value'] - reg_test['general_acceleration_add_value']
reg_test['building_acceleration'] = reg_test['building_acceleration_reduce_value'] - reg_test['building_acceleration_add_value']
reg_test['reaserch_acceleration'] = reg_test['reaserch_acceleration_reduce_value'] - reg_test['reaserch_acceleration_add_value']
reg_test['training_acceleration'] = reg_test['training_acceleration_reduce_value'] - reg_test['training_acceleration_add_value']
reg_test['treatment_acceleration'] = reg_test['treatment_acceleration_reduce_value'] - reg_test['treatment_acceleraion_add_value']

reg_test = reg_test.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)


'''
训练回归模型
'''
from lightgbm import LGBMRegressor
# =============================================================================
# data = data[data.prediction_pay_price > 0]
# =============================================================================

target = data['prediction_pay_price']
features = data.drop(['user_id','register_time','classification_label','prediction_pay_price'],axis=1)


reg = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=2000,min_child_weight=10,
                    subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)

'''
实际回归建模
'''

reg.fit(features, target, eval_set=[(features, target)], eval_metric='rmse',early_stopping_rounds=100)
reg_y_pre = reg.predict(reg_test)

reg_df = pd.DataFrame({ 
                        'user_id' : choose_test['user_id'],
                        'prediction_pay_price' : reg_y_pre,
                        })


final_df = pd.merge(clf_df_final, reg_df, on='user_id',how='left')
final_df = final_df.fillna(0)
final_df = final_df[['user_id','prediction_pay_price']]

final_df.to_csv('D:/999github/kaggle/sub_sample.csv', index=False)



