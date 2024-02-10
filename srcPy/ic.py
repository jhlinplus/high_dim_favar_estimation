
import numpy as np

class InformationCriteria():
    
    def __init__(self):
        pass
    
    @staticmethod
    def pic(ydata, xdata, B, Theta, rk):
    
        n, q = ydata.shape
        
        rss = np.linalg.norm(ydata - xdata@B.transpose(),'fro')**2
        sigma_hat = rss/(n*q)
        
        df_B = (B!=0).sum()
        penalty = sigma_hat * (np.log(n)/(n*q)*df_B + rk*(n+q)/(n*q)*np.log(n*q))
        
        picVal = sigma_hat + penalty
        
        return {'pic':picVal, 'sigma_hat':sigma_hat, 'penalty':penalty}
    
    @staticmethod
    def bic(ydata, xdata, B, penalty_type='ridge', lambda_coef=None):
        assert penalty_type in ['lasso','ridge']
        
        n = ydata.shape[0]
        rss = ((ydata - xdata@B.transpose())**2).sum(axis=0) ## rss for individual regression
        
        rss_sum = (np.log(rss)).sum()
        
        if penalty_type == 'lasso':
            df = (B!=0).sum()
        else:
            U, S, Vh = np.linalg.svd(xdata, full_matrices=False)
            numerator = S**2
            denominator = S**2 + np.array([lambda_coef]*len(S))
            df = ydata.shape[1] * (numerator/denominator).sum()
        return rss_sum + df*np.log(n)/n
    
    @staticmethod
    def bai_ic(ydata, rk):
    
        n,q = ydata.shape
        
        U, S, Vh = np.linalg.svd(ydata, full_matrices=False)
        S[rk:] = 0
        Smat = np.diag(S)
        Theta = U @ Smat @ Vh
        
        sigma_hat_sq = ((ydata - Theta)**2).sum(axis=0)/n

        return np.log(sigma_hat_sq.sum()/q) + r*(q+n)/(q*n)*np.log(n*q/(n+q))
        
        
