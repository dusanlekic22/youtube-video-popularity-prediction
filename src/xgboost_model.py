import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBClassifier

from data import *

if __name__ == '__main__':

    train, test = split_data(preprocessing_data(import_data()).drop(['title', 'description', 'tags'], axis=1))
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    # Print the value which is in x_train.iloc[:, np.r_[7:10, 18:33]].columns but not in x_train.columns
    print([(x_train.columns.get_loc(c), c) for c in x_train.columns if c in x_train])
    print(x_train.columns)
    dtrain = xgb.DMatrix(x_train, label=y_train)

    param = {'objective': 'binary:logistic', 'nthread': 4,
             'eval_metric': ['auc', 'mae', 'rmsle'],
             'colsample_bytree': 0.7872015451502918, 'learning_rate': 0.032176699545146153, 'max_depth': 14,
             'min_child_weight': 2, 'n_estimators': 297, 'subsample': 0.8259211487351279}

    dtest = xgb.DMatrix(x_test, label=y_test)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    bst.save_model('xgboost_models/0005.model')

    y_pred = bst.predict(dtest)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    auc_roc = roc_auc_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    micro_f1 = f1_score(y_test, predictions, average='micro')
    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('0001.model')  # load data
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("ROC: %.2f%%" % (auc_roc * 100.0))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("F1: %.2f%%" % (f1 * 100.0))
    xgb.plot_importance(bst)
    plt.show()
    # param_dist = {'n_estimators': stats.randint(150, 500),
    #               'learning_rate': stats.uniform(0.01, 0.07),
    #               'subsample': stats.uniform(0.3, 0.7),
    #               'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #               'colsample_bytree': stats.uniform(0.5, 0.45),
    #               'min_child_weight': [1, 2, 3]
    #               }
    #
    # clf_xgb = XGBClassifier(objective='binary:logistic')
    # clf = RandomizedSearchCV(clf_xgb, param_distributions=param_dist, n_iter=25, scoring='f1', error_score=0, verbose=3,
    #                          n_jobs=-1)
    # numFolds = 5
    # folds = KFold(n_splits=numFolds, shuffle=True)
    # X = preprocessing_data(import_data()).drop(['title', 'description', 'view_count','tags'], axis=1)
    # y = preprocessing_data(import_data())['view_count']
    # estimators = []
    # results = np.zeros(len(X))
    # score = 0.0
    # for train_index, test_index in folds.split(X):
    #     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    #     y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()
    #     clf.fit(X_train, y_train)
    #
    #     estimators.append(clf.best_estimator_)
    #     results[test_index] = clf.predict(X_test)
    #     score += f1_score(y_test, results[test_index])
    # score /= numFolds
    # print('Best hyperparameters:', clf.best_params_)