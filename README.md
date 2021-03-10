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

Diurnal pattern is observed in the modelled data. This results in strong autocorrelation and Partial autocorrelation. In order to model the data using INGARCH, a stationary process is needed. Hence, two steps were performed on the observations: 1) seasonality of the observations is modeled as the average daily observations over the course of a year, and; 2) remove the diurnal pattern (ex. 25% quantile of historical daily observations) from the lambda intensity in the process. The following listed three possible methods to remove the seasonality:

![image](https://user-images.githubusercontent.com/24922489/110702470-29c9f900-81b8-11eb-90a3-1b8e54c0e22c.png)

To measure the fitting performance of INGARCH model, the Bayesian information criterion (BIC) measure is used. The calculation is as the following:

![image](https://user-images.githubusercontent.com/24922489/110701970-7eb93f80-81b7-11eb-99f2-249598d3749f.png)

The measure evaluates the relationship between model’s log-likelihood and the number of parameters and training data used. A well performing model tends to have large log-likelihood, and smaller number of parameters and training data used. Hence, in this thesis project, models are ranked per their BIC such that the smallest is best.

Note that the model output was later converted into 3 regimes. Hence, to evaluate the classification accuracy, the following measure are used:

![image](https://user-images.githubusercontent.com/24922489/110702129-bde79080-81b7-11eb-9c61-f4cb5f407c84.png)


