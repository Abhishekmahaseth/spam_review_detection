from main import *
from sklearn.model_selection import train_test_split

# train a model
target_names = ['truthful', 'deceptive']
# splitting X and y into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.295, random_state=109)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# making predictions on the testing set
y_pred = gnb.predict(X_test)
# this provides Precision, Recall, Accuracy, F1-score
p, r, f, s = score(y_test, y_pred,average=None)
print("Precision:", p)
print("Recall:", r)
print("F1:", f)
print("Score:", s)
# this prints Precision, Recall, Accuracy, F1-score
print(classification_report(y_test, y_pred, target_names=target_names))

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# Visualize confusion matrix
# plot_confusion_matrix(gnb, X_test, y_test)

# Visualize ROC curve
# plot_roc_curve(gnb, X_test, y_test)

# Plotting the blobs for 2 catagories
# from sklearn.datasets import make_blobs
# X, y = make_blobs(100, 2, centers=2, random_state=5, cluster_std=1.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

plt.show()
