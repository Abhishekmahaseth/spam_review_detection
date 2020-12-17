from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
#from sklearn import precision_recall_fscore_support as varname

def MLPClassification(X_train,X_test,y_train,y_test):
    #1. Split the data into training and testing sets with a 80-20 split ratio
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #2. INitialize a model to the Random Forest Classifier from Sci-kit learn
    clf = MLPClassifier(max_iter=400, random_state=0)
    
    #3. Fit the model (train the model)
    clf.fit(X_train, y_train)
    
    #4. Now that the model is trained test the model
    y_pred = clf.predict(X_test)
    
    #5. Check the accuracy of the model
    #rfc_acc = clf.score(X_test,y_test)
    
    #6. Classification report: Precision, Recall and F1 Score calculation
    mlp_classification_report = classification_report(y_test, y_pred)
    print(mlp_classification_report)
    
    #7. Confusion matrix
    #mlp_conf_mat = confusion_matrix(y_test, y_pred)
    
    #8.Visualize confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    
    #9. Visualize ROC curve
    plot_roc_curve(clf, X_test, y_test)
    plt.show()                                
    
    return clf