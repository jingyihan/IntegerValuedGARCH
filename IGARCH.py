import matplotlib

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'figure.figsize': (20, 10),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15}
pylab.rcParams.update(params)

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import gammaln
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
from scipy.stats import poisson

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score


class INGARCH:
    
    def __init__(self, N, p, q, params_0, qtl, method):
        
        self.N = N 
        self.qtl = qtl
        
        self.p = p
        self.q = q
        self.params_0 = params_0
        self.gammas = params_0[1:self.p+2]
        self.deltas = params_0[self.p+2:]
        self.alpha = params_0[0]
        
        self.method = method
        
        self.lambdas = [[] for i in range(len(N))]
        self.mod_lambda = [[] for i in range(len(N))]
        self.logl = [[] for i in range(len(N))]
        
        self.a = [[] for i in range(len(N))]
        self.b = [[] for i in range(len(N))]
        
    def seasonality_factor(self, period):
        reps = np.ceil((len(self.N)+period)/len(self.qtl_1))
        self.qtl = np.tile(self.qtl_1, (1,int(reps)))
        self.qtl = self.qtl[0:(len(self.N)+period)]
        self.qtl = self.qtl[0]
        return
    
    def lambdas_a(self, i, j):
    
        # include the constant part of lambda
        lambdas_t = self.gammas[0]

        # ********************
        # include the sum_{n=1}^p N_{i-n} gamma_{i} term
        
        if(j>0):
            # compute the lagged point to go to, truncating at the sequence start
            j0_p = max(j-self.p, 0)
            if(j0_p>0):
                lambdas_t += np.sum(self.gammas[1:(j-j0_p)+1 ] * self.N[i][j-1:j-self.p-1:-1])
            else:
                lambdas_t += np.sum(self.gammas[1:(j-j0_p)+1 ] * self.N[i][j-1:None:-1])
            # ********************
            
            
            # ********************
            # include the sum_{n=1}^q lambda_{i-n} gamma_{i} term
            
            # compute the lagged point to go to, truncating at the sequence start
            j0_q = max(j-self.q, 0)
            if(j0_q>0):
                lambdas_t += np.sum(self.deltas[0:(j-j0_q) ] * self.lambdas[i][j-1:j-self.q-1:-1])
            else:
                lambdas_t += np.sum(self.deltas[0:(j-j0_q) ] * self.lambdas[i][j-1:None:-1])
            # ********************
        
        return lambdas_t

    def lambdas_b(self, i, j, qtl_test=None):

        # include the constant part of lambda
        if(qtl_test!=None):
            lambdas_t = max(1e-8, qtl_test * self.alpha) + self.gammas[0]
        else:
            lambdas_t = max(1e-8, self.qtl[i][j] * self.alpha) + self.gammas[0]

        # ********************
        # include the sum_{n=1}^p N_{i-n} gamma_{i} term
        
        if(j>0):
            # compute the lagged point to go to, truncating at the sequence start
            j0_p = max(j-self.p, 0)
            if(j0_p>0):
                lambdas_t += np.sum(self.gammas[1:(j-j0_p)+1 ] * self.N[i][j-1:j-self.p-1:-1])
            else:
                lambdas_t += np.sum(self.gammas[1:(j-j0_p)+1 ] * self.N[i][j-1:None:-1])
            # ********************
            
            
            # ********************
            # include the sum_{n=1}^q lambda_{i-n} gamma_{i} term
            
            # compute the lagged point to go to, truncating at the sequence start
            j0_q = max(j-self.q, 0)
            if(j0_q>0):
                lambdas_t += np.sum(self.deltas[0:(j-j0_q) ] * self.lambdas[i][j-1:j-self.q-1:-1])
            else:
                lambdas_t += np.sum(self.deltas[0:(j-j0_q) ] * self.lambdas[i][j-1:None:-1])
            # ********************
        
        return lambdas_t    
    
        
    def loglikelihood(self, params):
        
        self.gammas = params[1:self.p+2]
        self.deltas = params[self.p+2:]
        self.alpha = params[0]
        
        self.lambdas = [[] for i in range(len(self.N))]
        self.mod_lambda = [[] for i in range(len(self.N))]
        self.logl = [[] for i in range(len(self.N))]
        
        for i in range(len(self.N)):
            for j in range(len(self.N[i])):
            
                if self.method==1:
                    
                    self.lambdas[i].append(self.lambdas_a(i,j))
                    self.mod_lambda[i].append(self.lambdas[i][j] * max(1e-8, self.qtl[i][j] * self.alpha))
                    
                elif self.method==2:
                    
                    self.lambdas[i].append(self.lambdas_a(i,j))
                    self.mod_lambda[i].append(self.lambdas[i][j] + max(1e-8, self.qtl[i][j] * self.alpha))
                    
                else:
                    
                    self.lambdas[i].append(self.lambdas_b(i,j))
                    self.mod_lambda[i].append(self.lambdas[i][j])
                    
                
                self.logl[i].append(-self.mod_lambda[i][j] * self.alpha + self.N[i][j] * np.log(self.mod_lambda[i][j]) - gammaln(self.N[i][j]+1))
        
# =============================================================================
#         if(np.mod(self.iter,5)==0):
#             self.PlotFit(self.logl, self.lambdas, self.mod_lambda)
#         
#         print("-- iter " + str(self.iter))
#         print(-np.nansum(self.logl))
#         print(self.gammas)
#         print(self.deltas)
#         print("")
#         self.iter += 1
# =============================================================================
        
        sum_logl = 0
        for i in range(len(self.logl)):
            sum_logl += sum(self.logl[i])
        
        return -sum_logl
    
    def PlotFit(self, N, logl, lambdas, mod_lambda):
        
        logl_m = []
        lambdas_m = []
        mod_lambdas_m = []
        N_m = []
        
        for i in range(len(logl)):
            for j in range(len(logl[i])):
                logl_m.append(logl[i][j])
                lambdas_m.append(lambdas[i][j])
                mod_lambdas_m.append(mod_lambda[i][j])
                N_m.append(N[i][j])
        
        plt.subplot(1,3,1)
        plt.plot(lambdas_m,linewidth=0.5)
        
        plt.subplot(1,3,2)
        plt.plot(mod_lambdas_m,linewidth=0.5)
        plt.plot(N_m,linewidth=0.1)
        plt.ylim((0,100))
        
        plt.subplot(1,3,3)
        plt.plot(logl_m,linewidth=0.5)
        
        plt.show()    
        
        return 
    
    def PlotLarger(self, N, mod_lambda):
        
        mod_lambdas_m = []
        N_m = []
        
        for i in range(len(N)):
            for j in range(len(N[i])):
                mod_lambdas_m.append(mod_lambda[i][j])
                N_m.append(N[i][j])
        
        plt.plot(N_m,linewidth=0.4, color='#fb7d07', label='tick rates')
        plt.plot(mod_lambdas_m,linewidth=0.4, color='#276ab3', label='lambda with seasonality adjustment')
        #plt.plot(N_m,linewidth=0.1)
        plt.ylim((0,100))
        plt.legend()
        plt.show()
        return
    
    def fit(self, params):
        
# =============================================================================
#         if self.method == 1:
#             params[0] = 1
# =============================================================================
        A = np.ones(self.q+self.p+2)
        A[:2] = 0
        
        #self.seasonality_factor(0)
        bnds = [(0,1) for i in range(self.p+self.q+2)]
        bnds[0] = (0.4, 5)
        
        self.iter = 0
        res = minimize(self.loglikelihood, params, constraints=LinearConstraint(A, 0, 1), bounds=bnds, options={'xtol': 1e-8, 'disp': True})
        
        self.params = res.x
        
        self.gammas = res.x[1:self.p+2]
        self.deltas = res.x[self.p+2:]    
        self.alpha = res.x[0]
        
        lenN = [len(self.N[i]) for i in range(len(self.N))]
        self.bic = 0.5*(self.p+self.q+1) * np.log(sum(lenN)) + self.loglikelihood(res.x) 
        self.loglsum = -self.loglikelihood(res.x) 
        
        print("-- final ")
        print(self.loglikelihood(res.x))
        print(self.alpha)
        print(self.gammas)
        print(self.deltas)
        #self.PlotFit(self.N, self.logl, self.lambdas, self.mod_lambda)
        print("")

    
    def perdict(self, N_test, qtl_test, qtl_high, qtl_med, flag):
        
        self.regime = [[] for i in range(len(N_test))]
        self.prob = [[] for i in range(len(N_test))]
        
        self.lambdas_test = [[] for i in range(len(N_test))]
        self.mod_lambda_test = [[] for i in range(len(N_test))]
        self.logl_test = [[] for i in range(len(N_test))]
        
        # trace out lambdas for testing set using trained parameters
        for i in range(len(N_test)):
            for j in range(len(N_test[i])):
            
                if self.method==1:
                    
                    self.lambdas_test[i].append(self.lambdas_a(i,j))
                    self.mod_lambda_test[i].append(self.lambdas_test[i][j] * max(1e-8, qtl_test[i][j] * self.alpha ))
                    
                elif self.method==2:
                    
                    self.lambdas_test[i].append(self.lambdas_a(i,j))
                    self.mod_lambda_test[i].append(self.lambdas_test[i][j] + max(1e-8, qtl_test[i][j]* self.alpha))
                    
                else:
                    
                    self.lambdas_test[i].append(self.lambdas_b(i,j,qtl_test[i][j]))
                    self.mod_lambda_test[i].append(self.lambdas_test[i][j])
                    
                
                self.logl_test[i].append(-self.mod_lambda_test[i][j] + N_test[i][j] * np.log(self.mod_lambda_test[i][j]) - gammaln(N_test[i][j]+1))
                
                if flag:
                    # Regimes classification
                    high = 1-poisson.cdf(qtl_high[i][j],self.mod_lambda_test[i][j])
                    mid = poisson.cdf(qtl_high[i][j],self.mod_lambda_test[i][j])-poisson.cdf(qtl_med[i][j],self.mod_lambda_test[i][j])
                    low = poisson.cdf(qtl_med[i][j],self.mod_lambda_test[i][j])
                    
                    if(high==max(high,mid,low)):
                        self.regime[i].append(2)
                        self.prob[i].append(high)
                    elif(mid == max(high,mid,low)):
                        self.regime[i].append(1)
                        self.prob[i].append(mid)
                    else:
                        self.regime[i].append(0)
                        self.prob[i].append(low)
                        
            if not flag:           
                #Regimes classification:
                self.regime[i] = np.where(np.array(self.mod_lambda_test[i])-np.array(qtl_med[i])>=0, 1, 0)
                self.regime[i] = np.where(np.array(self.mod_lambda_test[i])-np.array(qtl_high[i])>=0, 2, self.regime[i])

                #print("regime: {}, probability: {}".format(self.regime[i][j], self.prob[i][j]))
        
        # Plot lambdas, modified lambdas, loglikelihood for testing sample
        self.PlotFit(N_test, self.logl_test, self.lambdas_test, self.mod_lambda_test)

    def accuracy(self, N_test, qtl_test, qtl_high, qtl_mid):
        
        #building actural regime
        self.actual = [[] for i in range(len(N_test))]
        
        for i in range(len(N_test)):
            self.actual[i] = np.where(np.array(N_test[i])-np.array(qtl_mid[i])>=0, 1, 0)
            self.actual[i] = np.where(np.array(N_test[i])-np.array(qtl_high[i])>=0, 2, self.actual[i])
        
        #join actual and predict regime together if in several consucative chunks
        self.actual_m = []
        self.regime_m = []
        if len(N_test) > 1:
            for i in range(len(self.actual)):
                for j in range(len(self.actual[i])):
                    self.actual_m.append(self.actual[i][j])
                    self.regime_m.append(self.regime[i][j])
        else:
            self.actual_m = self.actual[0]
            self.regime_m = self.regime[0]
        
        #plot modified lambdas with actual data
        self.PlotLarger(N_test, self.mod_lambda_test)
        
        #plot acutal and predict regimes
        plt.plot(self.actual_m, linewidth=0.5, color='#fb7d07')
        plt.plot(self.regime_m, linewidth=0.5, color='#276ab3')
        plt.show()
        
        #check number of predictions in each regimes and compare with acutual
        test_predict = pd.DataFrame(np.array(self.regime_m).T)
        print("Prediction:")
        print(test_predict.groupby(0).size())
        test_actual = pd.DataFrame(np.array(self.actual_m).T)
        print("Actual:")
        print(test_actual.groupby(0).size())
        
        #produce confusion matrix
        cm = confusion_matrix(test_predict, test_actual,labels=[0, 1, 2])
        cm_percent = cm.copy()
        cm_percent = cm_percent.astype(np.float64)
        cm_sum = cm_percent.sum(axis=1)
        for rs in range(cm.shape[1]):
            cm_percent[:,rs] = cm_percent[:,rs]/cm_sum
        self.cm_percent = cm_percent
        
        #classification error report
        #precision: TP/(TP+FP)
        #recall: TP/(TP+FN)
        #F1: 2*(precision*recall)/(precision+recall)
        print(classification_report(test_predict, test_actual, labels=[0, 1, 2]))
        self.percision = precision_score(test_predict, test_actual, labels=[0, 1, 2], average=None)
        self.recall = recall_score(test_predict, test_actual, labels=[0, 1, 2], average=None)
        self.accuracy_score = accuracy_score(test_predict, test_actual)
        
        FP = cm.sum(axis=1) - np.diag(cm)  
        FN = cm.sum(axis=0) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        # Sensitivity, hit rate, recall, or true positive rate
        self.TPR = TP/(TP+FN)
        # Fall out or false positive rate
        self.FPR = FP/(FP+TN)
        
        
        return
