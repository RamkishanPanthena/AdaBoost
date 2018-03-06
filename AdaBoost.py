# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:38:49 2018

@author: Krishna
"""
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# AdaBoost
class AdaBoost:
    def __init__(self, iterations):
        self.train_df = None
        self.test_df = None
        self.train_labels = None
        self.test_labels = None
        self.train_predictions = None
        self.test_predictions = None
        self.thresholds_dict = None #
        self.train_hx_dict = None
        self.columnname = None
        self.colsplitvalue = None
        self.epsilon = None
        self.alpha = None
        self.D = None
        self.trueClass = None
        self.falseClass = None
        self.train_H = None
        self.test_H = None
        self.iterations = iterations
        self.tr_err = None
        self.te_err = None
        self.rd_err = None
        self.fpr = None
        self.tpr = None
        
    # MainAlgo to fit the model
    def fit(self, train_X, train_y, test_X, test_y):
        self.train_df = train_X
        self.test_df = test_X
        self.train_labels = train_y[0].values
        self.test_labels = test_y[0].values
        self.train_H = np.zeros(shape = [1, len(train_X)], dtype = float)
        self.test_H = np.zeros(shape = [1, len(test_X)], dtype = float)
        
        self.initializeWeights(self.train_df)
        self.calc_thresholds()
        self.calc_hx()
        self.tr_err = []
        self.te_err = []
        self.rd_err = []
        for i in range(0, self.iterations):
            self.bestDecisionStump()                                        # Train a weak learner which is a single decision stump
            train_err = self.calc_error(self.train_labels, self.train_H[0]) # Calculate the train error
            test_err = self.calc_error(self.test_labels, self.test_H[0])    # Calculate the test error
            self.tr_err.append(train_err)                                   # Store the train,test and round errors obtained in every iteration
            self.te_err.append(test_err)
            self.rd_err.append(self.epsilon)
            self.fpr, self.tpr, thresholds = metrics.roc_curve(self.train_labels, self.train_H[0])
            auc = np.trapz(self.tpr, self.fpr)                              # Calculate the AUC value for the current iteration
            print("Round:", i+1, "Train_err:", train_err, "Test_err:", test_err, "AUC:", auc) # Print the train/test error and AUC values after each iteration
            self.updateWeights()                                            # Update weights based on the results of the current iteration
    
    # Initialize weights
    def initializeWeights(self, train):
        self.D = np.ones(shape = [1,len(train)], dtype = float)*1/len(train)
        
    # Update weights
    def updateWeights(self):
        D_old = self.D
        alpha = self.alpha
        labels = self.train_labels
        predictions = [self.train_predictions]
        
        D_new = D_old * np.exp(-alpha * labels * predictions)
        Z = np.sum(D_new)
        
        D_norm = D_new / Z
        
        self.D = D_norm
    
    # Compute all threshold values
    def calc_thresholds(self):
        df = self.train_df
        thresholds = dict()
        
        for column in range(self.train_df.shape[1]):
            values = df[column].values.tolist()
            values.sort()
            
            uniq_values = list(set(values))
            tmp_thresholds = []

            for i in range(len(uniq_values)-1):
                if i%50==0:
                    tmp_thresholds.append((uniq_values[i]+uniq_values[i+1])/2)
                
            thresholds[column] = tmp_thresholds

        self.thresholds_dict = thresholds
    
    # Compute predictions for all threshold values
    def calc_hx(self):
        df = self.train_df
        thresholds_dict = self.thresholds_dict
        train_hx = dict()
        
        for column in thresholds_dict:
            threshold = thresholds_dict[column]
            tmp = []

            for val in threshold:
                newdf1 = df.loc[df[column]<val]

                x1 = newdf1.loc[newdf1[0] == -1].count()[0]
                x2 = newdf1.loc[newdf1[0] == +1].count()[0]
                    
                if x1 > x2:
                    pr = [-1, 1]
                else:
                    pr = [1, -1]

                prediction = self.predict(df, column, val, pr)
                tmp.append([val, pr, prediction])
                
            train_hx[column] = tmp

        self.train_hx_dict = train_hx

    # Find the best predictor on which split needs to take place
    def bestDecisionStump(self):
        train_hx_dict = self.train_hx_dict
        
        bestepsilon = 0.0
        maxdiff = 0.0
        
        for column in train_hx_dict:
            threshold = train_hx_dict[column]
            
            for val in threshold:
                epsilon = self.calc_epsilon(self.D[0], self.train_labels, val[2])
                diff = abs(0.5 - epsilon)
                
                if diff > maxdiff:
                    maxdiff = diff
                    bestepsilon = epsilon
                    splitval = val[0]
                    splitcolumn = column
                    bestpred = val[2]
                    bestpr = val[1]
            
        self.columnname = splitcolumn
        self.colsplitvalue = splitval
        self.epsilon = bestepsilon
        self.alpha = math.log((1-bestepsilon)/bestepsilon)/2
        self.train_predictions = bestpred
        self.test_predictions = self.predict(self.test_df, self.columnname, self.colsplitvalue, bestpr)
        self.trueClass = bestpr[0]
        self.falseClass = bestpr[1]
        self.train_H = self.train_H + self.calc_Hx(self.alpha, self.train_predictions)
        self.test_H = self.test_H + self.calc_Hx(self.alpha, self.test_predictions)
        
    # Function to make prediction for a single row
    def predictRow(self, row, columnname, colsplitvalue, pr):
        col = columnname
        val = colsplitvalue
        global pred
        if row[col].values[0] < val:
            pred = pr[0]
        else:
            pred = pr[1]
            
        return pred
  
    # Function to make predictions for the entire dataset
    def predict(self, df, columnname, colsplitvalue, pr):
        prediction = []
        
        for i in range(len(df)):
            prediction.append(self.predictRow(df.iloc[[i]], columnname, colsplitvalue, pr))
        
        return prediction
        
    # Calculate epsilon to modify weights
    def calc_epsilon(self, D, actual, predictions):
        epsilon = 0
        for i in range(len(actual)):
            if actual[i] != predictions[i]:
                epsilon+=D[i]
         
        return epsilon

    # Calculate error
    def calc_error(self, actual, predictions):
        matchcount = 0
        for i in range(len(actual)):
            if (actual[i] > 0 and predictions[i] > 0) or (actual[i] < 0 and predictions[i] < 0):
                matchcount+=1
        
        return (1 - matchcount/len(actual))
    
    # Calculate Hx
    def calc_Hx(self, alpha, predictions):
        Hx = alpha * np.array(predictions)
        
        return Hx
        
    # Plot Train/Test Error
    def plot_train_test_error(self):
        plt.suptitle('Train/Test Error')
        plt.xlabel('Iteration Step'), plt.ylabel('Train/Test Error (Blue/Red Color)')
        plt.xlim(0, self.iterations), plt.ylim(0.05,.22)
        x = list(range(0, self.iterations))
        plt.plot(x, self.tr_err, color = "blue")
        plt.plot(x, self.te_err, color = "red")
        plt.show()
        
    # Plot ROC Curve
    def plot_ROC_curve(self):
        plt.suptitle('ROC Curve')
        plt.xlabel('Iteration Step'), plt.ylabel('AUC')
        plt.plot(self.fpr, self.tpr, color = "red")
        plt.show()
        
    # Plot Round Error
    def plot_round_error(self):
        plt.suptitle('Round Error')
        x = list(range(0, self.iterations))
        plt.xlabel('Iteration Step'), plt.ylabel('Round Error')
        plt.xlim(0, self.iterations), plt.ylim(0.2,.8)
        plt.plot(x, self.rd_err, color = "red")
        plt.show()