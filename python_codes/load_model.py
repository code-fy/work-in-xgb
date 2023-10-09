import xgboost as xgb

model = xgb.Booster()
model.load_model("./tmp_data_tmp_suibian_final_xgb.model_model_part-00000")
# model.dump_mode("./suibian_final_xgb.json")
print(model.get_fscore())
# model.dump_model('xgb_cancer.json')
data_dmatrix = xgb.DMatrix(data=[[0,1,1,0,1,0,100000]])
predict_result = model.predict(data_dmatrix)
print('predict_result: ', predict_result)
model.save_model("./suibian_final_xgb.model")
model.dump_model("./suibian_final_xgb.json")
model.feature_names = ["publisher_app_bundle","is_banner","device_lang","device_os","is_tablet","release_date","nfc_support","city"]
# xgb.plot_importance(model,max_num_features=8)
