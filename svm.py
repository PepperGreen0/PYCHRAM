#svm.py
import numpy as np
class SVM:
    def __init__(self, C=1.0):
        # C =error term
        self.C = C
        self.w = 0
        self.b = 0

    # Hinge Loss Function/Calculatinon
    def hingeloss(self,w,b,x,y):
        # Regularizer term
        reg = 0.5*(w*w)
        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i]*((np.dot(w,x[i]))+b)
            # calculating loss
            loss = reg + self.C*max(0,1-opt_term)
            return loss[0][0]
    def fit(self,X,Y,batch_size=100,learning_rate=0.001,epochs=1000):
        # The number of Samples in X
        number_of_features = X.shape[1]
        # The number of Samples in X
        number_of_samples = X.shape[0]
        c = self.C
        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)
        # Shuffling the samples randomly
        np.random.shuffle(ids)
        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []
        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w,b,X,Y)
            # Appendding all losses
            losses.append(l)
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0,number_of_samples,batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_initial, batch_initial+batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x]*(np.dot(w,X[x].T)+b)
                    if ti > 1:
                        gradw += 0
                        gradb += 0
                    else:
                        # Calculating the fradients
                        # w.r.t w
                        gradw += c*Y[x]*X[x]
                        # w.r.t b
                        gradb += c*Y[x]
                        # Updating weights and bias
                        w = w - learning_rate * w + learning_rate * gradw
                        b = b + learning_rate * gradb
                    self.w = w
                    self.b = b
                    return self.w,self.b,losses
    def predict(self, X: object):
        prediction=np.dot(X,self.w[0]+self.self.b) #w.x + b
        return np.sign(prediction)
