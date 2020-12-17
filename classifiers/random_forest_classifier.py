from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

def RandomForestClassification(X,y):
    #1. Split the data into training and testing sets with a 80-20 split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #2. INitialize a model to the Random Forest Classifier from Sci-kit learn
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    
    #3. Fit the model (train the model)
    clf.fit(X_train, y_train)
    
    #4. Now that the model is trained test the model
    y_pred = clf.predict(X_test)
    
    #5. Check the accuracy of the model
    #rfc_acc = clf.score(X_test,y_test)
    
    #6. Classification report: Precision, Recall and F1 Score calculation
    rfc_classification_report = classification_report(y_test, y_pred)
    
    #7. Confusion matrix
    rfc_conf_mat = confusion_matrix(y_test, y_pred)
    
    #8.Visualize confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    
    #9. Visualize ROC curve
    plot_roc_curve(clf, X_test, y_test)
    plt.show()                                
    
    return rfc_classification_report, rfc_conf_mat

def TunedRandomForestClassification(X,y):
    #1. Split the data into training and testing sets with a 80-20 split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #2. Initialize a model to the Random Forest Classifier from Sci-kit learn
    clf = RandomForestClassifier(random_state=0)
    
    
    #3. Initialize hyperparameter grid RandomizedSearchCV will search over
    grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["auto", "sqrt"],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4]}
    
    #4. Setup RandomizedSearchCV
    rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=20, # try 20 models total
                            cv=5, # 5-fold cross-validation
                            verbose=2) # print out results

    #5. Fit the RandomizedSearchCV version of clf
    rs_clf.fit(X_train, y_train);
    
    #6. Once the fit is complete take the best parameters for the different models tried
    rs_clf.best_params_
    
    #7. Make predictions with the best hyperparameters
    rs_y_preds = rs_clf.predict(X_test)
    
    #8. Classification report: Precision, Recall and F1 Score calculation
    rfc_classification_report = classification_report(y_test, rs_y_preds)
    
    #9. Confusion matrix
    rfc_conf_mat = confusion_matrix(y_test, rs_y_preds)
    
    #10.Visualize confusion matrix
    plot_confusion_matrix(rs_clf, X_test, y_test)
    
    #11. Visualize ROC curve
    plot_roc_curve(rs_clf, X_test, y_test)
    plt.show()                                
    
    return rfc_classification_report, rfc_conf_mat

def randomForest(X_train,X_test, y_train,y_test):
    
    #1. Initialize a model to the Random Forest Classifier from Sci-kit learn
    clf = RandomForestClassifier(random_state=0)
    
    
    #2. Initialize hyperparameter grid RandomizedSearchCV will search over
    grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["auto", "sqrt"],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4]}
    
    #3. Setup RandomizedSearchCV
    rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=20, # try 20 models total
                            cv=5, # 5-fold cross-validation
                            verbose=2) # print out results

    #4. Fit the RandomizedSearchCV version of clf
    rs_clf.fit(X_train, y_train);
    
    #5. Once the fit is complete take the best parameters for the different models tried
    rs_clf.best_params_
    
    #6. Make predictions with the best hyperparameters
    rs_y_preds = rs_clf.predict(X_test)
    
    #7. Classification report: Precision, Recall and F1 Score calculation
    rfc_classification_report = classification_report(y_test, rs_y_preds)
    print(rfc_classification_report)
    
    #8. Confusion matrix
    #rfc_conf_mat = confusion_matrix(y_test, rs_y_preds)
    
    #9.Visualize confusion matrix
    plot_confusion_matrix(rs_clf, X_test, y_test)
    
    #10. Visualize ROC curve
    plot_roc_curve(rs_clf, X_test, y_test)
    plt.show()                                
    
    return rs_clf