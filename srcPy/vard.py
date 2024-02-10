
import datetime
import numpy as np
from .customlasso import CustomLasso

class VARd():
    """
    Perform regularized VAR-d estimation
    \| Z - Zlag@A'\|_F^2 + lambdaA*\|A\|_1
    # (@param) d: number of lags of the VAR equation
    # (@param) lambdaA: penalty coefficient for A in the information equation
    """
    def __init__(
        self,
    ):
        pass
        
    def _utils_arrange_samples(self, zdata, d):
        """
        utility function for converting Zdata into samples based on lag specification
        Argv:
            - zdata (np.array) n-by-p raw input data
            - d (int) number of lags
        Return:
            - Z (np.array), (n-q)-by-p, i.e., Z_t
            - Zlag (np.array), (n-d)-by-q lag matrix with q = p*d, i.e., [Z_{t-1},...,Z_{t-d}]
        """
        Z = zdata.copy()[d:,:]
        Zlag_unstack = [zdata.copy()[(d-i):(-i),:] for i in range(1,d+1)]
        Zlag = np.concatenate(Zlag_unstack, axis=1)
        
        assert Z.shape[0] == Zlag.shape[0]      ## same number of rows
        assert Zlag.shape[1] == (zdata.shape[1] * d) ## correct number of columns
        
        return Z, Zlag
    
    def _utils_stack_A(self,A):
        """
        transform the flattened coef matrix for lags into a 3D array corresponding to the coef of each lag
        A_stacked[:,:,0] first lag; A_stacked[:,:,1] second lag, so on and so forth
        """
        p, q = A.shape
        d = int(q/p)
        A_stacked = np.zeros((p,p,d))
        for i in range(d):
            A_stacked[:,:,i] = A[:,(p*i):(p*(1+i))]
        return A_stacked
    
    def fitVARd(self, zdata, d = 1, lambdaA = 0.1, fit_intercept = False, stack = True, verbose=True):
        """
        solves a least-sqaure l1-regularized VARd that minimizes (1/2n)|| Z_n - Z_{n-1}A^\top ||_F^2 + lambdaA * |A|_1
        Argv:
            - zdata (np.array), n-by-p input data
            - d (int), number of lags
            - lambdaA (float), tuning parameter for regularized VAR
        Return:
            - A (np.array)
                if stack == True: p by p by d matrix, with A[:,:,i] corresponds to the transition matrix for lag (i+1)
                else: p by (pd), with columns (p*i):(p*(1+i)) corresponding to the transition coefficients of (i+1)-th lag
        """
        assert fit_intercept == False, 'currently fit_intercept=True is not supported; please center your data first'
        if verbose:
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Start VAR estimation')
        
        p = zdata.shape[1]
        Z, Zlag = self._utils_arrange_samples(zdata, d)
        A = np.zeros((p,p*d))
        
        model = CustomLasso(alpha=lambdaA, fit_intercept=fit_intercept)
        for j in range(p):
            model.fit(Zlag, Z[:,j])
            A[j,:] = model.coef_
        
        if stack:
            A = self._utils_stack_A(A)
        
        if verbose:
            print(f'>>> Done; A.shape={A.shape}')
            
        return A
