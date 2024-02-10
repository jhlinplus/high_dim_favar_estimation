
import datetime
import numpy as np
from .vard import VARd
from .customlasso import CustomLasso

class FAVAR(VARd):
    """
    - observation eqn obj:
    \| Y - X@B' - Theta\|_F^2 + lambdaB*\|B\|_1
    - VAR eqn obj, where Z is [F,X] stacked
    \| Z - Zlag@A'\|_F^2 + lambdaA*\|A\|_1
    (@param) IR: information criterion for extracting the factors, default to PC3
    - see Bai & Ng (2013, JoE) "Principal components estimation and identification of static factors" for the exact specification of PC1, PC2 and PC3
    
    
    # (@param) rk: rank of Theta in the information equation
    # (@param) d: number of lags of the VAR equation
    # (@param) lambdaB: penalty coefficient for B in the information equation
    # (@param) lambdaA: penalty coefficient for A in the information equation
    """
    def __init__(
        self,
        IR = 'PC3',
        max_iter = 500,
        tol = 1.0e-4,
    ):
        super(FAVAR, self).__init__()
        
        assert IR in ['PC1','PC2','PC3'], f'unrecoganized IR={IR}; choose among PC1, PC2, PC3'
        self.max_iter = max_iter
        self.tol = tol
        self.IR = IR
    
    def _info_objVal(self, ydata, xdata, B, Theta, lambdaB):
        
        loss = np.linalg.norm(ydata-xdata@B.transpose() - Theta, 'fro')**2/(2*xdata.shape[0])
        penalty = lambdaB * np.abs(B).sum()
        
        return loss + penalty
    
    def _extract_factors_and_loadings(self, Theta, rk):
        
        U, S, Vh = np.linalg.svd(Theta, full_matrices=False)
        V = Vh.transpose()
        if self.IR == 'PC3': ## F unrestricted
            estF = Theta[:,:rk]
            if rk == 1:
                estLambda_init = V[:,[0]] * S[0]/np.sqrt(Theta.shape[0])
                estLambda = estLambda_init/estLambda_init[0,0]
            else:
                estLambda_init = V[:,:rk] @ np.diag(S[:rk])/np.sqrt(Theta.shape[0])
                estLambda = estLambda_init @ np.linalg.inv(estLambda_init[:rk,:rk])
        else: # 'PC1' or 'PC2'
            if rk == 1:
                estF = np.sqrt(Theta.shape[0]) * U[:,[0]]
                estLambda = V[:,[0]] * S[0]/np.sqrt(Theta.shape[0])
            else:
                if self.IR == 'PC1': ## F orthorgonal, Lambda'Lambda is a diagonal with distinct entries
                    estF = np.sqrt(Theta.shape[0]) * U[:,:rk]
                    estLambda = V[:,:rk] @ np.diag(S[:rk])/np.sqrt(Theta.shape[0])
                else: ## PC2, ## F orthogonal, Lambda lower triangular
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
        A = self.fitVARd(zdata, d, lambdaA, fit_intercept=False, verbose=verbose)
        
        return {'estA': A, **info_output}
