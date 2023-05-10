from random import randint

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV

from data import *

if __name__ == '__main__':
    train, test = split_data(preprocessing_data(import_data()).drop(['title', 'description', 'tags'], axis=1))
    x_train, y_train, x_test, y_test = split_input_output(train, test)
    clf = svm.SVC(kernel='rbf', C=100, gamma=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("AUC:", metrics.roc_auc_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    # print f1 score
    print("F1:", metrics.f1_score(y_test, y_pred))
    wrong_predictions = test[y_pred != y_test.to_numpy().flatten()]

    #wrong_predictions[['likes', 'dislikes', 'comment_count', 'channel_view_count', 'channel_subscribe_count']].hist()
    plt.hist([wrong_predictions.loc[wrong_predictions['view_count'] == 0, 'likes'],
        wrong_predictions.loc[wrong_predictions['view_count'] == 1, 'likes']],
             stacked=True,
    label=['Unpopular', 'Popular'],
    edgecolor='white')
    plt.legend()
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #
    # # Use random search to find the best hyperparameters
    # rand_search = RandomizedSearchCV(clf,
    #                                  param_distributions=param_grid,
    #                                  refit=True,
    #                                  verbose=2)
    #
    # # Fit the random search object to the data
    # rand_search.fit(x_train, y_train)
    #
    # # Create a variable for the best model
    # best_svm = rand_search.best_estimator_
    #
    # # Print the best hyperparameters
    # print('Best hyperparameters:', rand_search.best_params_)