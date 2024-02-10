
import datetime
import numpy as np
from sklearn.linear_model import ElasticNet

from .favar import FAVAR
from .ic import InformationCriteria as IC

class AutoFAVAR(FAVAR):
    """
    inherited from FAVAR, but allow for supplying a sequence of tuning parameter to perform automatic selection based on the information criteria
    """
    def __init__(
        self,
        IR = 'PC3',
        max_iter = 500,
        tol = 1.0e-5,
    ):
        super(AutoFAVAR, self).__init__(IR, max_iter, tol)
    
    def autofitInfoEqn(self, ydata, xdata, rk_seq, lambdaB_seq, verbose=1):
        
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Stage I: running autofit for the Information Eqn over a lattice')
        
        pic_mtx = np.empty((len(rk_seq), len(lambdaB_seq)))
        for i in range(len(rk_seq)):
            for j in range(len(lambdaB_seq)):
                if verbose:
                    print(f'** i={i}/{len(rk_seq)}, j={j}/{len(lambdaB_seq)}')
                
                info_out = self.fitInfoEqn(ydata, xdata, rk=rk_seq[i], lambdaB=lambdaB_seq[j], verbose=0)
                pic_mtx[i,j] = IC.pic(ydata, xdata, info_out['estB'], info_out['estTheta'], rk_seq[i])['pic']
        
        ## find the smallest
        i_opt, j_opt = np.unravel_index(pic_mtx.argmin(), pic_mtx.shape)
        
        ## refit
        rk, lambdaB = rk_seq[i_opt], lambdaB_seq[j_opt]
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Done; r_best={rk} (index={i_opt}), lambdaB_best={lambdaB:.4f} (index={j_opt}); refitting ...')
    
        info_out = self.fitInfoEqn(ydata, xdata, rk=rk, lambdaB=lambdaB, verbose=1)
        return info_out, pic_mtx
        
    def autofitVARd(self, zdata, lambdaA_seq=None, d=1, fit_intercept=False, verbose=1):
        
        if lambdaA_seq is None:
            lambdaA_seq = np.arange(0.1,1,0.1) * np.sqrt(np.log(zdata.shape[1])/(zdata.shape[0]-d))
        
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Stage II: running autofit for the VAR Eqn over a grid')
        bic_seq = np.empty((len(lambdaA_seq),))
        Z, Zlag = self._utils_arrange_samples(zdata, d) ## used in calculating ic
        
        for i in range(len(lambdaA_seq)):
            if verbose:
                print(f'** i={i}/{len(lambdaA_seq)}')
            A = self.fitVARd(zdata, d, lambdaA_seq[i], fit_intercept=fit_intercept, stack=False, verbose=0)
            bic_seq[i] = IC.bic(Z, Zlag, A, penalty_type='lasso', lambda_coef=lambdaA_seq[i])
        
        i_opt = np.unravel_index(bic_seq.argmin(), bic_seq.shape)
        lambdaA = lambdaA_seq[i_opt]
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Done; lambdaA_best={lambdaA:.4f} (index={i_opt}); refitting ...')
        
        A = self.fitVARd(zdata, d, lambdaA, fit_intercept=fit_intercept, stack=True, verbose=1)
        return A, bic_seq
    
    def autofit(self, ydata, xdata, d, rk_seq, lambdaB_seq, lambdaA_seq, verbose=True):
        
        info_out, pic_mtx = self.autofitInfoEqn(ydata, xdata, rk_seq, lambdaB_seq, verbose=verbose)
        
        estF = info_out['estF']
        zdata = np.concatenate([estF, xdata], axis=1)
        
        A, bic_seq = self.autofitVARd(zdata, lambdaA_seq, d=d, fit_intercept=False, verbose=verbose)
        
        return {'estA': A, **info_out, 'pic': pic_mtx, 'bic': bic_seq}
