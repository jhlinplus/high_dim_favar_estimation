# Regularized Estimation of High-dimensional FAVAR Models

This is the official code repository for paper titled "Regularized estimation of high-dimensional factor-augmented vector autoregressive (FAVAR) models" (2020, Journal of Machine Learning Research), by **Jiahe Lin** and **George Michailidis**. 
- Link to paper: https://jmlr.csail.mit.edu/papers/volume21/19-874/19-874.pdf
- To cite this paper: Lin, J., Michailidis, G. (2020) Regularized estimation of high-dimensional factor-augmented vector autoregressive (FAVAR) models. Journal of Machine Learning Research, 21(117): 1–51.
    ```
    @article{lin2020regularized,
    title={Regularized estimation of high-dimensional factor-augmented vector autoregressive {(FAVAR)} models},
    author={Lin, Jiahe and Michailidis, George},
    journal={The Journal of Machine Learning Research},
    volume={21},
    number={1},
    pages={4635--4685},
    year={2020},
    publisher={JMLRORG}
    }
    ```

In this repository, we provide both `Python` and `R` implementation of the proposed two-stage methodology. 
- For `Python` version, see `demo.ipynb`
- For `R` version, see `R/example.R`

Note also that there is some difference in the behavior of `Lasso` from sklearn and `glmnet` in R