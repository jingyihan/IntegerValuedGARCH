# Integer-valued GARCH Model
Implemented the integer-valued GARCH model in Python. The key difference between GARCH model and integer-valued GARCH model is integer-valued GARCH assumed underlying distribution to be poisson distribution instead of normal distribution.

## 🔧 Key Functions
A class named INGARCH is created with the following key functions:
- Fit
- Predict
- Accuracy

## 📃 Model Details
An integer-valued analogue of the generalized autoregressive conditional heteroskedastic (GARCH) (p,q) model with Poisson deviates is used in this paper. An Integer-Valued GARCH, INGARCH(p,q), Process is defined as the following:

![image](https://user-images.githubusercontent.com/24922489/110701260-92b07180-81b6-11eb-85b6-32390a9bc7e8.png)

![image](https://user-images.githubusercontent.com/24922489/110701305-9fcd6080-81b6-11eb-97b5-734a67d5c5fb.png)
