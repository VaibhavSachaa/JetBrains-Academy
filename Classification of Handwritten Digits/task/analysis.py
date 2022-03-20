import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

(features_train, target_train), (features_test, target_test) = tf.keras.datasets.mnist.load_data()
features_train = features_train.reshape(features_train.shape[0], -1)

x_train, x_test, y_train, y_test = train_test_split(features_train, target_train, test_size=0.3,
                                                    random_state=40)

nrml = Normalizer()
x_train_norm = nrml.fit_transform(x_train)
x_test_norm = nrml.transform(x_test)


# the function
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    target_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, model.predict(features_test))
    # score = model.score(target_test, target_pred)
    # print(f'Model: {model}\nAccuracy: {score}\n')
    return score


knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=40)

print('K-nearest neighbours algorithm')

knn_params = [{'n_neighbors': [3, 4],
               'weights': ['uniform', 'distance'],
               'algorithm': ['auto', 'brute']}]

gcv_knn = GridSearchCV(estimator=knn, param_grid=knn_params, scoring='accuracy', n_jobs=-1)
gcv_knn.fit(x_train_norm, y_train)

print("best estimator: {}".format(gcv_knn.best_estimator_))

knn_best = gcv_knn.best_estimator_
knn_best.fit(x_train_norm, y_train)
#gcv_knn_best = GridSearchCV(estimator=knn_best, param_grid=knn_params, scoring='accuracy', n_jobs=-1)
#gcv_knn_best.fit(x_test_norm, y_test)
#print("acurracy:", int(gcv_knn_best.score(x_test_norm, y_test)))
print("accuracy: {}".format(knn_best.score(x_test_norm, y_test)))

print("\nRandom forest algorithm")

forest_params = [{'n_estimators': [300, 500],
                  'max_features': ['auto', 'log2'],
                  'class_weight': ['balanced', 'balanced_subsample']}]

gcv_forest = GridSearchCV(estimator=rf, param_grid=forest_params, scoring='accuracy', n_jobs=-1)
gcv_forest.fit(x_train_norm, y_train)

print("best estimator: {}".format(gcv_forest.best_estimator_))

rf_best = gcv_forest.best_estimator_
rf_best.fit(x_train_norm, y_train)

print("accuracy: {}".format(rf_best.score(x_test_norm, y_test)))
