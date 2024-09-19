#Binary Classification
import clf
import svm
from sklearn.datasets import make_classification
import numpy as np

from svmmulticls import SVM

#Load the datasset
np.random.seed(1)
X, y = make_classification(n_samples=2500,n_features=5,
                           n_redundant=0, n_informative=5,
                           n_classes=2, class_sep=0.3)
# Test Implemented SVM
svm = SVM(kernel='rbf', k=1)
svm.fit(X, y,eval_train=True)
y_pred,_=svm.predict(X)
print(f" Accuracy: {np.sum(y==y_pred)/y.shape[0]}") #0.9108
# Test with Scikit
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=1, gamma=1)
clf.fit(X, y)
y_pred = clf.predict(X)
print(f"Accuracy: {sum(y==y_pred)/y.shape[0]}") #0.9108

