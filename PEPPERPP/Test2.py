#Multiclass Classification
from sklearn.datasets import make_classification
import svm
import clf
import numpy as np
from svmmulticls import SVM
# Load the dataset
np.random.seed(1)
X, y = make_classification (n_samples=500, n_features=2,
                            n_redundant=0, n_informative=2,
                            n_classes=4, n_clusters_per_class=1,
                            class_sep=0.3)

# Test SVM
svm = SVM(kernel='rbf', k=4)
svm.fit(X, y,eval_train=True)
y_pred = sum.predict(X)
print(f"Accuracy: {np.sum(y==y_pred)/y.shape[0]}") # 0.65
#Test with Scikit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
clf = OneVsRestClassifier(SVC(kernel='rbf', C=1, gamma=4)).fit(X, y)
y_pred = clf.predict(X)
print(f"Accuracy: {sum(y==y_pred)/y.shape[0]}") # 0.65