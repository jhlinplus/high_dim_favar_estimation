
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class CustomLasso():
    """
    customize Lasso class that mimics the standardize behavior in glmnet
    Note that this only provides the fit part so that the coefficients are estimated in the same way
    """
    def __init__(
        self,
        alpha,
        fit_intercept,
        standardize = True,
        tol = 1.0e-5,
    ):
        self.standardize = standardize
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y):
        
        if not self.standardize:
            self.Lasso = Lasso(alpha=self.alpha,fit_intercept=self.fit_intercept,tol=self.tol)
            self.Lasso.fit(X,y)
            self.coef_ = self.Lasso.coef_
        else:
            
            scX = StandardScaler().fit(X)
            Xstd = scX.transform(X)
            
            #scY = StandardScaler().fit(y.reshape(-1,1))
            #ystd = scY.transform(y.reshape(-1,1)).ravel()
            
            self.Lasso = Lasso(alpha=self.alpha,fit_intercept=False,tol=self.tol)
            self.Lasso.fit(Xstd,y)
            
            if not self.fit_intercept:
                self.coef_ = self.Lasso.coef_ * np.sqrt(1.0/scX.var_)
            else:
                coef = np.zeros_like(self.Lasso.coef_)
                coef[1:] = self.Lasso.coef[1:] * np.sqrt(1.0/scX.var_)
                coef[0] = self.Lasso.coef_[0] - np.sum(scX.mean_*coef[1:])
                self.coef_ = coef
        
        return
