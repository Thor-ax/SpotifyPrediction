import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from Preprocess import Preprocess
from sklearn.model_selection import train_test_split
from Classifier import Classifiers

train_data  = pd.read_csv("spotify_dataset_train.csv")
test_data  = pd.read_csv("spotify_dataset_test.csv")

preprocess = Preprocess(train_data)

preprocessTest = Preprocess(test_data)

(X_data, y_data) = preprocess.PreprocessDs(True)


X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size=0.25,random_state=0)

Clf = Classifiers(X_train, y_train)

preprocessTest.PreprocessDs(False)
# testDs = preprocessTest.getDs()
# preprocess.frequency(testDs)

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

# Clf.drawTree(X_test, y_test)

pred = Clf.randomForest(X_test, y_test)

pred =preprocess.inverse(pred)
column = ['genre']
genre=pd.DataFrame(columns=column,data=pred)
genre.to_csv('genre.csv',index = False) 