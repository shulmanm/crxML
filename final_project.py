'''
Matthew Shulman
CIS 192 Final Project

Supervised Machine Learning with Credit Approval Data

Source: http://archive.ics.uci.edu/ml/datasets/Credit+Approval
'''
from sklearn import metrics
import csv
import random
import pandas as pd
from MLModel import MLModel


droplist = set()

#generator to get number of random rows from dataframe later
def generate_rows(numget, total):
    count = 0
    past = []
    global droplist
    while count < numget:
        a = random.choice(range(0, total))
        if a not in droplist and a not in past:
            past.append(a)
            yield a
            count+=1

def csv_to_data():
    # load the CSV file
    data = pd.read_csv('./crx.csv', header=None)

    #drop if missing values
    global droplist
    for index, row in data.iterrows():
        for i in range(0,16):
            if row[i] is '?':
                droplist.add(index)

    data = data.drop(list(droplist))

    # Pluses and minuses
    classification = data[15]

    # Numerical columns numbers: 1, 2, 7, 10, 13, 14
    data = data.drop(0, 1)
    data = data.drop(3, 1)
    data = data.drop(4, 1)
    data = data.drop(5, 1)
    data = data.drop(6, 1)
    data = data.drop(8, 1)
    data = data.drop(9, 1)
    data = data.drop(11, 1)
    data = data.drop(12, 1)
    features = data.drop(15, 1)

    #turn +, - into 1, 0

    s = pd.Series()
    for num, sign in classification.iteritems():
        if sign is '+':
            s.set_value(num, 1)
        else:
            s.set_value(num, 0)
    classification = s
    return features, classification

def make_sets():
    '''splits data into training/testing'''

    features, classification = csv_to_data()

    #make 2 copies features, 2 copies classifiactions for data split
    featuresc1 = features.copy()
    classificationc1 = classification.copy()
    featuresc2 = features.copy()
    classificationc2 = classification.copy()

    global droplist
    rownums = [x for x in range(0, features.shape[0]) if x not in droplist]

    #split 80% training data, 20% testing data
    for row in generate_rows(int(features.shape[0]*0.8), features.shape[0]):
        rownums.remove(row)

        featuresc2 = featuresc2.drop(row)
        classificationc2 = classificationc2.drop(row)

    featuresc1 = featuresc1.drop(rownums)
    classificationc1 = classificationc1.drop(rownums)

    #rename for clarity
    features_train = featuresc1
    features_test = featuresc2
    classification_train = classificationc1
    classification_test = classificationc2
    return features_train, classification_train, features_test, classification_test


def classify(ml):
    '''
    Trains a classifier on the features_train data.
    Used to classify the features_test data.
    Return the result.
    '''
    features_train, classification_train, features_test, classification_test = make_sets()

    model = MLModel(ml).getClass()
    model = model.fit(features_train, classification_train)
    return model.predict(features_test)


def analyze(ml):
    '''Analyzes the accuracy of the classifier against the test data.
    Prints out the table of its precision, recall and f1-score.'''

    features_train, classification_train, features_test, classification_test = make_sets()
    predicted = classify(ml)
    expected = classification_test
    print(metrics.classification_report(expected, predicted))



#choose ML Algorithm as argument for analyze
#Must choose 'DT', 'GB', 'KN', 'LC', 'LR'
def main():
    analyze('DT')

if __name__ == "__main__":
	main()
