import matplotlib.pyplot as plt
import xgboost as xgb
from data import *

if __name__ == '__main__':

    train, test = split_data(preprocessing_data(import_data()).drop(['title', 'description'], axis=1))
    x_train, y_train, x_test, y_test = split_input_output(train, test)
    # Print the value which is in x_train.iloc[:, np.r_[7:10, 18:33]].columns but not in x_train.columns
    print([(x_train.columns.get_loc(c), c) for c in x_train.columns if c in x_train])
    print(x_train.columns)
    dtrain = xgb.DMatrix(x_train, label=y_train)

    param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic', 'nthread': 4,
             'eval_metric': ['auc', 'mae', 'rmsle']}

    dtest = xgb.DMatrix(x_test, label=y_test)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    bst.save_model('0005.model')

    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('0001.model')  # load data

    xgb.plot_importance(bst)
    plt.show()