# Integer-valued GARCH Model
Implemented the integer-valued GARCH model in Python. The key difference between GARCH model and integer-valued GARCH model is integer-valued GARCH assumed underlying distribution to be poisson distribution instead of normal distribution.

## 🔧 Key Functions
A class named INGARCH is created with the following key functions:
- Fit
- Predict
- Accuracy

## 📃 Model Details
An integer-valued analogue of the generalized autoregressive conditional heteroskedastic (GARCH) (p,q) model with Poisson deviates is used in this paper. An Integer-Valued GARCH, INGARCH(p,q), Process is defined as the following:

![image](https://user-images.githubusercontent.com/24922489/110701676-19654e80-81b7-11eb-935c-ff8f8c92c53a.png)


INGARCH model parameters are estimated using Maximum Log-Likelihood approach and constrained numerical optimization is used to obtain parameter estimates. The log-likelihood is calculated with the follow function:

![image](https://user-images.githubusercontent.com/24922489/110701619-03f02480-81b7-11eb-8bd2-f62d56a3ecbb.png)

