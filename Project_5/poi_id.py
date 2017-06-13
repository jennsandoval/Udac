#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import pylab as pl

sys.path.append("../tools/")

from copy import copy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from tester import test_classifier
from time import time
from matplotlib import pyplot as plt
from sklearn import cross_validation


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 #'deferral_payments',
                 #'total_payments',
                 #'loan_advances',
                 #'bonus',
                 #'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 #'other',
                 #'long_term_incentive',
                 #'restricted_stock',
                 #'director_fees',
                 'to_messages',
                 #'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 #'shared_receipt_with_poi'
                 ]

print "Starting amount of features: ", len(features_list) - 1



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Amount of Data Points included in the Dataset: ", len(data_dict)


## Print the list of persons included in the dataset
x = []
for person in data_dict.keys():
    x.append(person)
print "Persons included in the Dataset: ", x

## Check for NaN values for each feature included in the dataset

all_features = data_dict['METTS MARK'].keys()

missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1

print('Amount of NaN values for each Feature:')
for feature in all_features:
    print("%s: %d" % (feature, missing_values[feature]))

### Task 2: Remove outliers

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)

print "Amount of Data Points after the Outliers are removed from the Dataset: ", len(data_dict)

    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_feature_list = copy(features_list)


#Bonus-Salary Ratio for employees
for employee, features in my_dataset.iteritems():
    if features['bonus'] == "NaN" or features['salary'] == "NaN":
        features['bonus_salary_ratio'] = "NaN"

    else: features['bonus_salary_ratio'] = float(features['bonus']/features['salary'])


#To POI Message Ratio

for person in my_dataset.values():
    if person['from_poi_to_this_person'] == "NaN" or person['to_messages'] == "NaN":
        person['from_poi_to_other_percentage'] = "NaN"

    else: person['from_poi_to_other_percentage'] = float(person['from_poi_to_this_person']/person['to_messages'])


#From POI Message Ratio

for person in my_dataset.values():
    if person['from_this_person_to_poi'] == "NaN" or person['from_messages'] == "NaN":
        person['from_this_person_to_poi_percentage'] = "NaN"

    else: person['from_this_person_to_poi_percentage'] = float(person['from_this_person_to_poi']/person['from_messages'])


# get K-best features

num_features = 8 

def get_k_best(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])

    print "{0} best features:{1}\n".format(k, k_best_features.keys()), scores
    return k_best_features


best_features = get_k_best(my_dataset, my_feature_list, num_features)

my_feature_list = ['poi'] + best_features.keys()



###Add new features that were created to features list and print the updated Features_List

my_feature_list.extend(['bonus_salary_ratio', 'from_poi_to_other_percentage', 'to_poi_to_other_percentage'])
    
print "Updated Features List: ", my_feature_list[1:]



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Scal features using min max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Logistic Regression

from sklearn.linear_model import LogisticRegression

regression_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 10**-21, C = 10**18, penalty = 'l2', random_state = 42))])


#K Means Clustering

from sklearn.cluster import KMeans

kmeans_clf = KMeans(n_clusters=2, tol=0.001)

#Random Forest

from sklearn.ensemble import RandomForestClassifier

randomforest_clf = RandomForestClassifier()

#Adaboost

from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(algorithm='SAMME')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


# Create, fit, and make predictions with grid search

from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys

def evaluate_clf(clf, features, labels, num_iters=1500, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))

    print "precision:{}".format(mean(precision))
    print "recall:{}".format(mean(recall))
    return mean(precision), mean(recall)


evaluate_clf(regression_clf, features, labels)
evaluate_clf(kmeans_clf, features, labels)
evaluate_clf(randomforest_clf, features, labels)
evaluate_clf(adaboost_clf, features, labels)

#Best Performer
clf = regression_clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
