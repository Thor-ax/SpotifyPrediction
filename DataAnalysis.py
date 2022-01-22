from Preprocess import Preprocess
from sklearn.model_selection import train_test_split
from Classifier import Classifiers
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

df  = pd.read_csv("spotify_dataset_subset.csv")

preprocess = Preprocess(df)

preprocess.infos()

# no nan values
preprocess.nbNanValues()

#preprocess.corr()

(X_data, y_data) = preprocess.preprocessDsForPopularityPrediction()

#pca
x_data = preprocess.pca(X_data)

preprocess.infos()

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size=0.25,random_state=0)


Clf = Classifiers(X_train, y_train)


def test():
    #decisionTree
    Clf.decisionTree(X_test, y_test)
    print()

    #randomForest
    Clf.randomForest(X_test, y_test)
    print()

    #KNN
    Clf.knn(X_test, y_test)
    print()

    #SVM
    Clf.svm(X_test, y_test)
    print()

    #Gradient boosting
    Clf.gradientBoosting(X_test, y_test)
    print()

    #Logistic regression
    Clf.logisticRegression(X_test, y_test)
    print()

    #Extra tree
    Clf.extraTree(X_test, y_test)
    print()

    #Voting
    Clf.voting(X_test, y_test)
    print()



Clf.bestMLPParams(X_test, y_test)





