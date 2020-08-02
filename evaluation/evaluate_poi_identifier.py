#!/usr/bin/python3


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

# data = featureFormat(data_dict, features_list)
# labels, features = targetFeatureSplit(data)



### your code goes here 
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

train_split, test_split, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state=42)

### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(train_split, train_labels)
pred = clf.predict(test_split)
accuracy = accuracy_score(pred, test_labels)
print(accuracy)

from collections import Counter
count = Counter(pred)
print(count[1])

test_numbers = len(test_split)
print(test_numbers)

count_for_real_poi = Counter(test_labels)
count_for_real_poi_1=  count_for_real_poi[1]
print('if all the predicted labels are 0', (test_numbers - count_for_real_poi_1)/test_numbers)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print('Recall_score: ', recall_score(test_labels, pred))
print('Precision_score: ',precision_score(test_labels, pred))
