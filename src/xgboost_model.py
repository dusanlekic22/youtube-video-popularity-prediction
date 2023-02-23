import matplotlib.pyplot as plt
import xgboost as xgb
from data import *

if __name__ == '__main__':

    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    dtrain = xgb.DMatrix(x_train[['likes', 'dislikes', 'comment_count', 'trending_time']], label=y_train)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc'}

    dtest = xgb.DMatrix(x_test[['likes', 'dislikes', 'comment_count', 'trending_time']], label=y_test)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    bst.save_model('0001.model')

    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('0001.model')  # load data

    xgb.plot_importance(bst)
    plt.show()