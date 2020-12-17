from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

def AdaBoostClassification(X,y):
    #1. Split the data into training and testing sets with a 80-20 split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #2. INitialize a model to the Ada Boost Classifier from Sci-kit learn
    clf = AdaBoostClassifier(random_state=0)
    
    #3. Fit the model (train the model)
    clf.fit(X_train, y_train)
    
    #4. Now that the model is trained test the model
    y_pred = clf.predict(X_test)
    
    #5. Check the accuracy of the model
    #ada_acc = clf.score(X_test,y_test)
    
    #6. Classification report: Precision, Recall and F1 Score calculation
    ada_classification_report = classification_report(y_test, y_pred)
    
    #7. Confusion matrix
    ada_conf_mat = confusion_matrix(y_test, y_pred)
    
    #8.Visualize confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    
    #9. Visualize ROC curve
    plot_roc_curve(clf, X_test, y_test)
    plt.show()                                
    
    return ada_classification_report, ada_conf_mat

def TunedAdaBoostClassification(X,y):
    #1. Split the data into training and testing sets with a 80-20 split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    #2. Initialize a model to the Ada Boost Classifier from Sci-kit learn
    clf = AdaBoostClassifier(random_state=0)
    
    
    #3. Initialize hyperparameter grid GridSearchCV will search over
    grid = {"n_estimators": [100, 200, 300],
            }

    #4. Setup GridSearchCV
    ada_clf = GridSearchCV(estimator=clf,
                            param_grid=grid,
                            scoring= 'accuracy',
                            n_jobs=-1, 
                            cv=5,
                            verbose=2) 

    #5. Fit the GridSearchCV version of clf
    ada_clf.fit(X_train, y_train);
    
    #6. Once the fit is complete take the best parameters for the different models tried
    ada_clf.best_params_
    
    #7. Make predictions with the best hyperparameters
    ada_y_preds = ada_clf.predict(X_test)
    
    #8. Classification report: Precision, Recall and F1 Score calculation
    ada_classification_report = classification_report(y_test, ada_y_preds)
    
    #9. Confusion matrix
    ada_conf_mat = confusion_matrix(y_test, ada_y_preds)
    
    #10.Visualize confusion matrix
    plot_confusion_matrix(ada_clf, X_test, y_test)
    
    #11. Visualize ROC curve
    plot_roc_curve(ada_clf, X_test, y_test)
    plt.show()                                
    
    return ada_classification_report, ada_conf_mat

def adaBoost(X_train,X_test, y_train,y_test):
    
    #1. Initialize a model to the Ada Boost Classifier from Sci-kit learn
    clf = AdaBoostClassifier(random_state=0)
    
    
    #2. Initialize hyperparameter grid GridSearchCV will search over
    grid = {"n_estimators": [100, 200, 500, 1000],
            }

    #3. Setup GridSearchCV
    ada_clf = GridSearchCV(estimator=clf,
                            param_grid=grid,
                            scoring= 'accuracy',
                            n_jobs=-1, 
                            cv=5,
                            verbose=2) 

    #4. Fit the GridSearchCV version of clf
    ada_clf.fit(X_train, y_train);
    
    #5. Once the fit is complete take the best parameters for the different models tried
    ada_clf.best_params_
    
    #6. Make predictions with the best hyperparameters
    ada_y_preds = ada_clf.predict(X_test)
    
    #7. Classification report: Precision, Recall and F1 Score calculation
    ada_classification_report = classification_report(y_test, ada_y_preds)
    print(ada_classification_report)
    
    #8. Confusion matrix
    #ada_conf_mat = confusion_matrix(y_test, ada_y_preds)
    
    #9.Visualize confusion matrix
    plot_confusion_matrix(ada_clf, X_test, y_test)
    
    #10. Visualize ROC curve
    plot_roc_curve(ada_clf, X_test, y_test)
    plt.show()                                
    
    return ada_clf