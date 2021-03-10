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

![image](https://user-images.githubusercontent.com/24922489/110701842-4d407400-81b7-11eb-995b-dad47891533e.png)

To measure the fitting performance of INGARCH model, the Bayesian information criterion (BIC) measure is used. The calculation is as the following:

![image](https://user-images.githubusercontent.com/24922489/110701970-7eb93f80-81b7-11eb-99f2-249598d3749f.png)

The measure evaluates the relationship between model’s log-likelihood and the number of parameters and training data used. A well performing model tends to have large log-likelihood, and smaller number of parameters and training data used. Hence, in this thesis project, models are ranked per their BIC such that the smallest is best.

Note that the model output was later converted into 3 regimes. Hence, tTo evaluate the classification accuracy, the following measure are used:

![image](https://user-images.githubusercontent.com/24922489/110702129-bde79080-81b7-11eb-9c61-f4cb5f407c84.png)


