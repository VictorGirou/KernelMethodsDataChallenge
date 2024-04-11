import numpy as np

class Ridge_Regression:

    def __init__(self,kernel,C = 1.0):
        self.kernel = kernel
        self.n = len(Xtr)
        self.alpha = np.zeros((10,self.n))
        self.C = C

    def fit(self,X,y):

        K = self.kernel(self.Xtr,self.Xtr)
        N = K.shape[0]
        
        A11 = (2/N)*K.T@K + self.C*K
        A12 = np.reshape(np.ones(N),(N,1))
        A21 = (1/N)*np.ones(N).T@K
        A22 = np.array([2*N])
        A = np.block([[A11,A12],[A21,A22]])
        b = np.block([[(2/N)*K.T@y],[np.reshape(np.mean(y,axis = 0),(1,y.shape[1]))]])

        sol = np.linalg.solve(A,b)
        self.b = sol[-1,:]
        self.alpha =  sol[0:N,:]
        self.x = X

        
    ### Implementation of the separting function $f$ 
    def regression_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(x,self.x)
        return K@self.alpha
    
    def predict(self, X):
        return 2*(self.regression_function(X)+np.expand_dims(self.b,axis=0) > 0)-1
        