import numpy as np
from cvxopt import matrix, solvers
from scipy import optimize
from scipy.sparse import csc_matrix, csr_array

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        K = matrix(self.kernel(X, X))
        diag_y = matrix(np.diag(y))
        N = len(y)

        
        
        Q = matrix(diag_y * K * diag_y)
        p = matrix(-np.ones(N))

        G = matrix(np.block([[np.eye(N)], [-np.eye(N)]]))
        h = matrix(np.concatenate([self.C*np.ones(N), np.zeros(N)]))

        A = matrix(np.reshape(y, (1, -1)), (1, N), 'd')
        b = matrix(np.zeros(1))

        self.alpha = np.array(solvers.qp(Q,p,G,h,A,b)['x']).flatten()

        ## Assign the required attributes

        indices_margin = np.logical_and((0 < y*self.alpha).flatten(), (y*self.alpha < self.C).flatten())

        self.margin_points = X[indices_margin, :]#A matrix with each row corresponding to a point that falls on the margin
        self.support = np.array([X[i, :] for i in range(N) if self.alpha[i]!=0])
        self.alpha = diag_y @ self.alpha
        
        self.b = np.mean((y - (self.kernel(X, X) @ self.alpha))[indices_margin]) #offset of the classifier
        
        self.alpha = self.alpha[self.alpha!=0]

        self.norm_f = self.alpha.T @ (self.kernel(self.support, self.support) @ self.alpha)   #RKHS norm of the function f


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.alpha
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1