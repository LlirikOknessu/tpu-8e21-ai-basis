'''from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=3000,
                          early_stopping_rounds=100,
                          grow_policy='Depthwise',
                          depth=8,
                          loss_function=RMSLE(),
                          cat_features=CAT_COLS,
                          random_state=RS,
                          l2_leaf_reg=1,
                          learning_rate=0.03,
                          verbose=10,
                          eval_metric=RMSLE_val())

params = {'l2_leaf_reg':[1,4,8],
          'learning_rate': [0.03,0.5,0.1]
          'depth':[6,8,10]
         }
grid_search_res = model.grid_search(params, full_features['items'][FTS_COLS], full_features['items'].target, train_size=0.8)
'''