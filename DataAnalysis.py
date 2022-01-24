from Preprocess import Preprocess
from sklearn.model_selection import train_test_split
from Classifier import Classifiers
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

df  = pd.read_csv("spotify_dataset_subset.csv")

preprocess = Preprocess(df)

preprocess.nbNanValues()

preprocess.infos()

# no nan values
preprocess.nbNanValues()

preprocess.corr()

(X_data, y_data) = preprocess.preprocessDsForPopularityPrediction()
#pca => to accelerate tests => remove for better results
X_data = preprocess.pca(X_data)

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

test()