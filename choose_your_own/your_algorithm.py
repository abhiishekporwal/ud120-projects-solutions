#!/usr/bin/python3

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#KNN
clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_knn.fit(features_train, labels_train)
pred_knn = clf_knn.predict(features_test)
acc_knn = accuracy_score(pred_knn, labels_test)
print('accuracy for KNeighborsClassifier: ', acc_knn)

#AdaBoost
clf_adaboost  = AdaBoostClassifier(n_estimators = 10, random_state = 0)
clf_adaboost.fit(features_train, labels_train)
pred_adaboost = clf_adaboost.predict(features_test)
acc_adaboost = accuracy_score(pred_adaboost, labels_test)
print('accuracy for AdaBoostClassifier: ', acc_adaboost)


#RandomForest
clf_rfc  = RandomForestClassifier(n_estimators = 15, random_state = 0)
clf_rfc.fit(features_train, labels_train)
pred_rfc = clf_rfc.predict(features_test)
acc_rfc = accuracy_score(pred_rfc, labels_test)
print('accuracy for RandomForestClassifier: ', acc_rfc)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
