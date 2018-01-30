# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:07:48 2018

@author: Krishna
"""
import pandas as pd


# Load the entire dataset into a dataframe and split into train and test
def spambase(filename):
    df = pd.read_csv(filename, sep = ',', header = None)
    
    n = df.shape[1] - 1
    updlabel = df[n].values
    for i in range(len(updlabel)):
        if updlabel[i] == 0:
            updlabel[i] = -1

    df[n] = updlabel

    train_X = pd.DataFrame()
    train_y = pd.DataFrame()
    test_X = pd.DataFrame()
    test_y = pd.DataFrame()
    
    for i in range(len(df)):
        if i%5 != 0:
            train_X = pd.concat([train_X, df.iloc[[i]]])
            train_y = pd.concat([train_y, df.iloc[[i]][n]])
        else:
            test_X = pd.concat([test_X, df.iloc[[i]]])
            test_y = pd.concat([test_y, df.iloc[[i]][n]])
    
    del train_X[n]
    del test_X[n]

    return (train_X, train_y, test_X, test_y)