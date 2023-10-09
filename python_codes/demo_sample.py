from sklearn.model_selection import train_test_split
from pandas import DataFrame

from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost.spark import SparkXGBClassifier

from xgboost import plot_tree
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X = DataFrame(X)
y = DataFrame(y)
# breast_cancer.feature_names的名字中带有空格，会报错。
X.columns = breast_cancer.feature_names
X.columns = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12', 'l13', 'l14',
             'l15', 'l16', 'l17', 'l18', 'l19', 'l20', 'l21', 'l22', 'l23', 'l24', 'l25',
             'l26', 'l27', 'l28', 'l29', 'l30', ]
X
# X.columns = ['l1']
X = X[["l1"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
'''
X_train.columns在第一段代码中也已经设置过了。
特别需要注意：列名字中不能有空格。
'''
ceate_feature_map(X_train.columns)
clf = XGBClassifier(
    n_estimators=1,  
    learning_rate=0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)

model_sklearn = clf.fit(X_train, y_train)
at  = X_test[:1]
print(at.to_dict())
y_sklearn = clf.predict_proba(at)
print(y_sklearn)
model_sklearn.save_model('./cancer.json')
model = xgb.Booster()
model.load_model("./model_file_name.json")
model.dump_mode("./model_file_name_demo.json")
print(model.get_fscore())
# model.dump_model('xgb_cancer.json')
# data_dmatrix = xgb.DMatrix(data=at)
# predict_result = model.predict(data_dmatrix)
# print('predict_result: ', predict_result)
model.save_model("./model_file_name.model")
# model.load_model("./bin_model")
# predict_result = model.predict(data_dmatrix)
# print('predict_result: ', predict_result)

