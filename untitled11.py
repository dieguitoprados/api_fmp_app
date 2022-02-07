# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 01:44:06 2021

@author: diegu
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA
from pandas_datareader import data

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


start=dt.datetime(2016, 1, 1)
end=dt.datetime.now
stocks=['CME', 'SI', 'NVDA', 'TWTR', 'FB', 'TSLA', 'LCID']


class api_yahoo():
    
    def __init__(self, start, end, symbol):
        self.start=start
        self.end=end
        self.stocks=symbol
        self.df=None
        self.change=None
        self.shape=None
        self.mean=pd.DataFrame()
        self.std=pd.DataFrame()
        self.skewness=pd.DataFrame()
        self.kurt=pd.DataFrame()
        self.jarque_bera_stat=pd.DataFrame()
        self.p_value=pd.DataFrame()
        self.is_normal=pd.DataFrame()
        self.sharpe=pd.DataFrame()
        self.var_95=pd.DataFrame()
        self.cvar_95=pd.DataFrame()
        self.median=pd.DataFrame()
        
    def load (self):    

        self.df=data.DataReader(self.stocks, 'yahoo', self.start, self.end )
        self.df=self.df.dropna()
        # self.df.Close.CME
        # df['return_close']=df['Close']
        self.shape=self.df.shape[0]
        # for c in stocks
        self.change=self.df.Close/self.df.Close.shift(1) - 1
        self.change=self.change.dropna()
        # df.get_level_values("second")
        
        # return self.change, self.df, self.shape

    def compute(self):
        self.stats=pd.DataFrame(index=['mean','median',\
                                           'std','skewness','kurtosis',\
                                           'jarque bera','p-value', 'isnormal',\
                                               'sharpe','var_95%','Cvar_95%'])

        for elements in self.change:
            self.mean=np.mean(self.change[elements])
            self.std = np.std(self.change[elements])
            self.skewness = skew(self.change[elements])
            self.kurt = kurtosis(self.change[elements]) # excess kurtosis
            self.jarque_bera_stat = self.shape/6*(self.skewness**2 + 1/4*self.kurt**2)
            self.p_value = 1 - chi2.cdf(self.jarque_bera_stat, df=2)
            self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
            self.sharpe = self.mean /self.std * np.sqrt(252) # annualised
            self.var_95 = np.percentile(self.change[elements],5)
            self.cvar_95 = np.mean(self.change[elements][self.change[elements] <= self.var_95])
            # percentile_25 = self.percentile(25)
            self.median = np.median(self.change[elements])
            # self.stats=pd.DataFrame(index=['mean','median',\
            #                                'std','skewness','kurtosis',\
            #                                'jarque bera','p-value', 'isnormal',\
            #                                    'sharpe','var 95%','Cvar 95%'])
                                    # columns=[self.mean,self.median,self.std,self.skewness,\
                                    #          self.kurt,self.jarque_bera_stat,\
                                    #          self.p_value,self.is_normal,\
                                    #          self.sharpe,self.var_95,self.cvar_95])
            self.stats[elements]=[self.mean,self.median,self.std,self.skewness,\
                                  self.kurt,self.jarque_bera_stat,self.p_value,\
                                  self.is_normal,self.sharpe,self.var_95,self.cvar_95]
            
        # percentile_75 = self.percentile(75)
            print('----------------------------------------')
            print(elements+' median is '+str(self.median))
            print(elements+' mean is '+str(self.mean))
            print(elements+' std is '+str(self.std))
            print(elements+' skew is '+str(self.skewness))
            print(elements+' kurt is '+str(self.kurt))
            print(elements+' jarque_bera_stat is '+str(self.jarque_bera_stat))
            print(elements+' p_value is '+str(self.p_value))
            print(elements+' is_normal is '+str(self.is_normal))
            print(elements+' sharpe is '+str(self.sharpe))
            print(elements+' var_95 is '+str(self.var_95))
            print(elements+' cvar_95 is '+str(self.cvar_95))
            
    def loadtimeseries(self):
            
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        for elements in self.df.Close:
            elements = (self.df.Close[elements]/self.df.Close[elements][0]*100).plot(kind='line', ax=ax, grid=True, label=elements)
        # ax2 = df['price1'].plot(kind='line', x=df['date'], ax=ax, grid=True,\
        #                       color='red', secondary_y=True, label=)
            elements.legend(loc=2)
        # ax2.legend(loc=1)

        plt.show()
    def loadtimeseries_log(self):
            
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        for elements in self.df.Close:
            elements = (self.df.Close[elements]/self.df.Close[elements][0]*100).plot(kind='line', ax=ax, grid=True, label=elements)
        # ax2 = df['price1'].plot(kind='line', x=df['date'], ax=ax, grid=True,\
        #                       color='red', secondary_y=True, label=)
            elements.legend(loc=2)
        # ax2.legend(loc=1)
        plt.yscale('log')
        plt.show()
        
    def plot_histogram(self):
        for elements in self.stocks:
            plt.figure()
            plt.hist(self.change[elements],bins=100)
            plt.title(elements+' Daily Returns')
            plt.xlabel(str(self.stats[elements]))
            plt.show()

    
# df.return_close=change

    # df1=returns_CME.dropna()
    # plt.figure()
    # plt.hist(returns_CME,bins=100)
    # plt.title('Market Data '+'CME')
    # plt.show()

# dff=pd.read_csv("C:\\Users\\diegu\\Downloads\\12-28-2021_Amplify_BLOK_Holdings.csv")
# dff=dff.dropna()
# dff=dff.drop(dff.index[[3]])
# dff=dff.reset_index(drop=True)
# df.index
# df1=pd.DataFrame()
# df1.index=df.index
# df1['close']=df['Close']
