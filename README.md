## Regularized estimation of high-dimensional FAVAR models

A factor-augmented VAR model has two parts: 

(information/calibration eqn) \\Y_t = BX_t + \Lambda F_t + e_t \\
(VAR eqn) \\ F_t = A_{11}F_{t-1} + A_{12}X_{t-1} + w^F_t \\ and \\X_t = A_{21}F_{t-1} + A_{22}X_{t-1} + w^X_t \\

\\X_t,Y_t\\ are observed, \\F_t\\ is latent. 

The code in this repository considers estimating such a model in the high-dimensional setting, where both the regression coefficient matrix \\B\\ and the transition matrix \\A\\ are sparse. 

Reference: https://arxiv.org/abs/1912.04146
