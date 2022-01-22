from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



class Classifiers:

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def decisionTree(self, X_test, y_test):
        dt = DecisionTreeClassifier()
        dt.fit(self.X_data, self.y_data)
        score = dt.score(X_test, y_test)
        MSE = mean_squared_error(y_test, dt.predict(X_test))
        f1 = f1_score(y_test,dt.predict(X_test),average='weighted')
        print('-----')
        print('Decision tree')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def neuralNetwork(self, X_test, y_test, hls = 100):
        mlp = MLPClassifier(hidden_layer_sizes=hls, max_iter=500)
        mlp.fit(self.X_data, self.y_data)
        score = mlp.score(X_test, y_test)
        MSE = mean_squared_error(y_test, mlp.predict(X_test))
        f1 = f1_score(y_test,mlp.predict(X_test),average='weighted')
        print('-----')
        print('MLP')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def bestMLPParams(self, X_test, y_test):
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100))

        params = {
            'hidden_layer_sizes': [(100, 100, 100), (125, 125, 125)]  # Every combination you want to try
        }

        gscv = GridSearchCV(clf, params, verbose=1)

        gscv.fit(np.array(self.X_data), np.array(self.y_data))
        print(gscv.best_params_)

        predicted_values = gscv.predict(X_test)


        score = accuracy_score(y_test, predicted_values)

        print(score)


    def randomForest(self, X_test, y_test):
        rf = RandomForestClassifier(random_state=0)
        rf.fit(self.X_data, self.y_data)
        score = rf.score(X_test, y_test)
        MSE = mean_squared_error(y_test, rf.predict(X_test))
        f1 = f1_score(y_test,rf.predict(X_test),average='weighted')
        print('-----')
        print('Random forest')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def knn(self, X_test, y_test):
        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(self.X_data, self.y_data)
        score = knn.score(X_test, y_test)
        MSE = mean_squared_error(y_test, knn.predict(X_test))
        f1 = f1_score(y_test,knn.predict(X_test),average='weighted')
        print('-----')
        print('KNN')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def svm(self, X_test, y_test):
        lin_clf = svm.LinearSVC()
        lin_clf.fit(self.X_data, self.y_data)
        score = lin_clf.score(X_test, y_test)
        f1 = f1_score(y_test,lin_clf.predict(X_test),average='weighted')
        MSE = mean_squared_error(y_test, lin_clf.predict(X_test))
        print('-----')
        print('SVM')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def bagging(self, X_test, y_test):
        clf = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators = 15).fit(self.X_data, self.y_data)
        score = clf.score(X_test, y_test)
        f1 = f1_score(y_test,clf.predict(X_test),average='weighted')
        MSE = mean_squared_error(y_test, clf.predict(X_test))
        print('-----')
        print('bagging')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def stacking(self, X_test):
        estimators = [('rf', RandomForestClassifier(random_state=0)),
                      ('sgd', make_pipeline(StandardScaler(), SGDClassifier(penalty='l1')))]
        estimators2 = [('rf', RandomForestClassifier(random_state=0)),
                      ('sgd', KNeighborsClassifier(n_neighbors=20))]
        estimators3 = [('rf', RandomForestClassifier(random_state=0)),
                       ('sgd', KNeighborsClassifier(n_neighbors=20)), ('et', ExtraTreesClassifier(n_estimators=200))]
        clf = StackingClassifier(estimators=estimators2, final_estimator=RandomForestClassifier()).fit(
            self.X_data, self.y_data)
        print('-----')
        print('prediction with Stacking')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict

    def gradientBoosting(self, X_test, y_test):
        gbc = GradientBoostingClassifier(random_state=0)
        gbc.fit(self.X_data, self.y_data)
        score = gbc.score(X_test, y_test)
        f1 = f1_score(y_test,gbc.predict(X_test),average='weighted')
        MSE = mean_squared_error(y_test, gbc.predict(X_test))
        print('-----')
        print('Gradient boosting')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def logisticRegression(self, X_test, y_test):
        clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
        clf1.fit(self.X_data, self.y_data)
        score = clf1.score(X_test, y_test)
        f1 = f1_score(y_test,clf1.predict(X_test),average='weighted')
        MSE = mean_squared_error(y_test, clf1.predict(X_test))
        print('-----')
        print('Logistic regression')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def extraTree(self, X_test, y_test):
        etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
        etc.fit(self.X_data, self.y_data)
        score = etc.score(X_test, y_test)
        f1 = f1_score(y_test,etc.predict(X_test),average='weighted')
        MSE = mean_squared_error(y_test, etc.predict(X_test))
        print('-----')
        print('Extra Tree')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def voting(self, X_test, y_test):
        clf1 = ExtraTreesClassifier(n_estimators=100, random_state=0)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = GradientBoostingClassifier(random_state=0)
        clf4 = KNeighborsClassifier(n_neighbors=20)
        eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('knn', clf3), ('gb', clf4)], voting='hard')
        eclf1 = eclf1.fit(self.X_data, self.y_data)
        f1 = f1_score(y_test,eclf1.predict(X_test),average='weighted')
        score = eclf1.score(X_test, y_test)
        MSE = mean_squared_error(y_test, eclf1.predict(X_test))
        print('-----')
        print('Voting')
        print('-----')
        print("f1_score = "+str(round(100 * f1, 2)) + "%")
        print("Mean accuracy = " + str(round(100 * score, 2)) + "%")
        print("Mean squared error = " + str(round(MSE, 2)))

    def bestRandomForestParameters(self, X_test, y_test):
        randomForestClassifier = RandomForestClassifier(random_state=0)
        parameters = {
                      'n_estimators':[50, 100, 200],
                      'max_depth':[int (x) for x in np.linspace(10,500,10)]
                }

        clf = GridSearchCV(randomForestClassifier, parameters, cv=5)
        clf.fit(self.X_data, self.y_data)
        score = clf.score(X_test,y_test)
        score_test = mean_squared_error(y_test, clf.predict(X_test))
        print("parameters",clf.cv_results_['params'])
        print("scores",clf.cv_results_['mean_test_score'])
        print("best score", clf.best_score_)
        print("best parameters", clf.best_params_)
