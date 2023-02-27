import matplotlib.pyplot as plt
import xgboost as xgb
from data import *

if __name__ == '__main__':

    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)
    print(x_train.iloc[:, np.r_[7:10, 14:30]].columns)
    dtrain = xgb.DMatrix(x_train.iloc[:, np.r_[7:10, 14:30]], label=y_train)

    param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': ['auc', 'mae', 'rmsle']}

    dtest = xgb.DMatrix(x_test.iloc[:, np.r_[7:10, 14:30]], label=y_test)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    bst.save_model('0001.model')

    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('0001.model')  # load data

    xgb.plot_importance(bst)
    plt.show()