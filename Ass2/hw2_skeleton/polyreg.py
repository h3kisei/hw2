'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None
        self.mean = None
        self.std = None


    def polyfeatures(self, X, degree):

        #TODO
        expandedArr = []
        for x in range(0, X.size):
            curArr = []
            for y in range(0, degree):
                curArr.append(X[x]**(1 + y))
            expandedArr.append(curArr)
        return expandedArr
        

    def fit(self, X, y):

        #TODO
        # convert X into a n * d array of polynomial features of 
        # degree d
        XExpanded = self.polyfeatures(X, self.degree)

        XExpandedNP = np.array(XExpanded)

        # get std and mean for training data (to be used for testing as well)
        std = np.std(XExpandedNP, axis=0)
        mean = np.mean(XExpandedNP, axis=0)
        self.std = std
        self.mean = mean

        # standardize data
        XExpandedNP = (XExpandedNP - mean) / std

        # add the zero-th order feature row (i.e. x_0 = 1)
        XExpandedNP = np.c_[np.ones((XExpandedNP.shape[0],1)), XExpandedNP]

        # fit
        n,d = XExpandedNP.shape
        d = d - 1
        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(XExpandedNP.T.dot(XExpandedNP) + regMatrix).dot(XExpandedNP.T).dot(y);
        
    def predict(self, X):

        # TODO
        XExpanded = self.polyfeatures(X, self.degree)
        XExpandedNP = np.array(XExpanded)

        # standardize data based on training means and stds
        XExpandedNP = (XExpandedNP - self.mean) / self.std

        # add the zero-th order feature row (i.e. x_0 = 1)
        XExpandedNP = np.c_[np.ones((XExpandedNP.shape[0],1)), XExpandedNP]

        return XExpandedNP.dot(self.theta)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in xrange(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)
    