
import datetime
import numpy as np
from .customlasso import CustomLasso

class FAVAR():
    """
    - observation eqn obj:
    \| Y - X@B' - Theta\|_F^2 + lambdaB*\|B\|_1
    - VAR eqn obj, where Z is [F,X] stacked
    \| Z - Zlag@A'\|_F^2 + lambdaA*\|A\|_1
    (@param) IR: information criterion for extracting the factors, default to IC3
    - see Bai & Ng (2013, JoE) "Principal components estimation and identification of static factors" for the exact specification of IC1, IC2 and IC3
    
    
    # (@param) rk: rank of Theta in the information equation
    # (@param) d: number of lags of the VAR equation
    # (@param) lambdaB: penalty coefficient for B in the information equation
    # (@param) lambdaA: penalty coefficient for A in the information equation
    """
    def __init__(
        self,
        IR = 'IC3',
        max_iter = 500,
        tol = 1.0e-4,
    ):
        assert IR in ['IC1','IC2','IC3'], f'unrecoganized IR={IR}; choose among IC1, IC2, IC3'
        self.max_iter = max_iter
        self.tol = tol
        self.IR = IR
    
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
    
    def fitRegularizedVAR(self, zdata, d = 1, lambdaA = 0.1, fit_intercept = False, stack = True, verbose=True):
        """
        solves a least-sqaure l1-regularized VARd that minimizes (1/2n)|| Z_n - Z_{n-1}A^\top ||_F^2 + lambdaA * |A|_1
        Argv:
            - zdata (np.array), n-by-p input data
            - d (int), number of lags
            - lambdaA (float), tuning parameter for regularized VAR
        Return:
            - A (np.array), p by p by d coefficient matrix, with A[:,:,i] corresponds to the transition matrix for lag (i+1)
        """
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
    
    def _info_objVal(self, ydata, xdata, B, Theta, lambdaB):
        
        loss = np.linalg.norm(ydata-xdata@B.transpose() - Theta, 'fro')**2/(2*xdata.shape[0])
        penalty = lambdaB * np.abs(B).sum()
        
        return loss + penalty
    
    def _extract_factors_and_loadings(self, Theta, rk):
        
        U, S, Vh = np.linalg.svd(Theta, full_matrices=False)
        V = Vh.transpose()
        if self.IR == 'IC3': ## F unrestricted
            estF = Theta[:,:rk]
            if rk == 1:
                estLambda_init = V[:,[0]] * S[0]/np.sqrt(Theta.shape[0])
                estLambda = estLambda_init/estLambda_init[0,0]
            else:
                estLambda_init = V[:,:rk] @ np.diag(S[:rk])/np.sqrt(Theta.shape[0])
                estLambda = estLambda_init @ np.linalg.inv(estLambda_init[:rk,:rk])
        elif self.IR == 'IC1': ## F orthorgonal, Lambda'Lambda is a diagonal with distinct entries
            if rk == 1:
                estF = np.sqrt(Theta.shape[0]) * U[:,[0]]
                estLambda = V[:,[0]] * S[0]/np.sqrt(Theta.shape[0])
            else:
                estF = np.sqrt(Theta.shape[0]) * U[:,:rk]
                estLambda = V[:,:rk] @ np.diag(S[:rk])/np.sqrt(Theta.shape[0])
        else: ## F orthogonal, Lambda lower triangular
            if rk == 1:
                estF = np.sqrt(Theta.shape[0]) * U[:,[0]]
                estLambda = V[:,[0]] * S[0]/np.sqrt(Theta.shape[0])
            else:
                estF_init = np.sqrt(Theta.shape[0]) * U[:,:rk]
                estLambda_init = V[:,:rk] @ np.diag(S[:rk])/np.sqrt(Theta.shape[0])
                Q, _ = np.linalg.qr(estLambda_init[:rk, :rk])
                estF = estF_init @ Q
                estLambda = est_Lambda @ Q
                
        return estF, estLambda

    
    def fitInfoEqn(self, ydata, xdata, rk, lambdaB, verbose=0):
        """
        estimate the information equation
        """
        if verbose:
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Start Information Equation Estimation')
        ### initialization ###
        Theta = np.zeros_like(ydata)
        B = np.zeros((ydata.shape[1], xdata.shape[1]))
        
        itx, CONVERGE, f_valSeq = 0, False, []
        f_initial = self._info_objVal(ydata, xdata, B, Theta, lambdaB)
        if verbose > 1:
            print(f'>> f_initial = {f_initial:.4f}')
        
        ### iterate ###
        lassoEngine = CustomLasso(alpha=lambdaB, fit_intercept=False)
        while (not CONVERGE):
            
            ## update B
            for j in range(ydata.shape[1]):
                lassoEngine.fit(xdata, (ydata-Theta)[:,j])
                B[j,:] = lassoEngine.coef_
            
            fB = self._info_objVal(ydata, xdata, B, Theta, lambdaB)
            fB_delta = fB - f_initial if itx == 0 else fB - f_valSeq[itx-1]
            ## update Theta using singlar value thresholding
            U, S, Vh = np.linalg.svd( ydata - xdata @ B.transpose(), full_matrices=False)
            S[rk:] = 0
            diag_S = np.diag(S)
            Theta = U @ diag_S @ Vh
            ## tracking/convergence
            fval = self._info_objVal(ydata, xdata, B, Theta, lambdaB)
            fTheta_delta = fval - fB
            f_delta = fB_delta + fTheta_delta
            if verbose > 1:
                print(f'>> iter={itx}, fB_delta={fB_delta:.4f}, fTheta_delta={fTheta_delta:.4f}, f_delta={f_delta:.4f}')
            CONVERGE = np.abs(f_delta) < self.tol
            itx += 1
            f_valSeq.append(fval)
            
            if itx > self.max_iter:
                print(f'!! Forcing iterations to stop @ {self.max_iter}')
        
        if CONVERGE:
            if verbose:
                print(f'>>> Converged @ iter = {itx}; f_terminal = {f_valSeq[-1]:.4f}')
        
        estF, estLambda = self._extract_factors_and_loadings(Theta, rk)
        return {'estTheta': Theta, 'estF': estF, 'estLambda':estLambda, 'estB': B}
    
    def fit(self, ydata, xdata, d, rk, lambdaB, lambdaA, verbose=True):
        
        ## estimating information eqn
        info_output = self.fitInfoEqn(ydata, xdata, rk, lambdaB, verbose=verbose)
        estF = info_output['estF']
        ## estimating VAR equation
        zdata = np.concatenate([estF, xdata], axis=1)
        A = self.fitRegularizedVAR(zdata, d, lambdaA, fit_intercept=False, verbose=verbose)
        
        return {'estA': A, **info_output}
