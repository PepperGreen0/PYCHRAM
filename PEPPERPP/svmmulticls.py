import numpy as np  #for basic operations over arrays
from scipy.spatial import distance  #to compute the Gaussian kernel
import cvxopt       #to solve the dual opt, problem
import copy         #to copy numpy arrays
class SVM:
    linear = lambda  x, xp, c=0: x @ xp.T
    polynomail = lambda x, xp, Q=5: (1 + x @ xp.T)**Q
    rbf = lambda  x, xp, y=10: np.exp(-y*distance.cdist(x, xp,'sqeuclidean'))
    kernel_funs = {'linear':linear,'polynomial':polynomail,'rbf':rbf}

    def __init__(self, kernel='rbf', C=1, k=2):
        # set the hyperparameters
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C  # regularization parameter
        self.k = k  # kernel parameter

        # trainning data and support vectors
        self.X, y = None, None
        self.aps = None

        # for multiclass classification
        self.multiclass = False
        self.clfs = []
    def fit(self, X, y, eval_train=False):
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        # relabel if needed
        if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
        # ensure y has dimensions Nx1
        self.y = y.reshape(-1, 1).astype(np.double)  # Has to be a column vector
        self.X = X
        N = X.shape[0]
        # compute the kernel over all possible pairs of (x. x*) in the data
        self.K = self.kernel(X, X, self.k)
        # For 1/2 X^T Px + g^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N), np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N, 1)), np.ones((N, 1)) * self.C)))
        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.aps = np.array(sol["x"])
        # Maps into support vectors
        self.is_sv = ((self.aps > 1e-3) & (self.aps <= self.C)).squeeze()
        self.margin_sv = np.argmax((1e-3 < self.aps) & (self.aps < self.C - 1e-31))
        if eval_train:
            print(f"Finished training with accuracy{self.evaluate(X, y)}")
    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))  # number of classes
        # for each pair of classes
        for i in range(self.k):
            # get the data for the pair
            Xs, Ys = X, copy.copy(y)
            # change the labels to -1 and 1
            Ys[Ys!=i], Ys[Ys==i] = -1, +1
            # fit the classifier
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        if eval_train:
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def multi_evaluate(self, X, y):
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)

    def predict(self, X_t):
        if self.multiclass: return self.multi_predict(X_t)
        # compute (xs, ys)
        xs, ys = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        # find support vectors
        aps,y, X = self.aps[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
        # compute the second term
        b = ys - np.sum(aps * y * self.kernel(X, xs, self.k), axis=0)
        # compute the score
        score = np.sum(aps * y * self.kernel(X,X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score
    def multi_predict(self, X):
        preds = np.zeros((X.shape[0], self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[: i] = clf.predict(X)
            # get the argmax and the corresponding score
        return np.argmax(preds, axis=1)
    def evaluate(self, X, y):
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)


