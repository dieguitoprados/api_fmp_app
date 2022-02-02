# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:59:02 2021

@author: diegu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:15:32 2021

@author: Meva
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)
from scipy.optimize import minimize


class distribution_manager(): 
    
    def __init__(self, inputs):
        self.inputs = inputs # distribution_inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None
        self.vec_returns = None
        self.mean = None
        self.std = None
        self.skew = None
        self.kurtosis = None # excess kurtosis
        self.jarque_bera_stat = None # under normality of self.vec_returns this distributes as chi-square with 2 degrees of freedom
        self.p_value = None # equivalently jb < 6
        self.is_normal = None
        self.sharpe = None
        self.var_95 = None
        self.cvar_95 = None
        self.percentile_25 = None
        self.median = None
        self.percentile_75 = None
        
        
    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self
        
        
    def load_timeseries(self):
        
        # data_type = self.inputs['data_type']
        data_type = self.inputs.data_type
        
        if data_type == 'simulation':
            
            nb_sims = self.inputs.nb_sims
            dist_name = self.inputs.variable_name
            degrees_freedom = self.inputs.degrees_freedom
            
            if dist_name == 'normal':
                x = np.random.standard_normal(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'exponential':
                x = np.random.standard_exponential(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'uniform':
                x = np.random.uniform(0,1,nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'student':
                x = np.random.standard_t(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
            elif dist_name == 'chi-square':
                x = np.random.chisquare(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
       
            self.nb_rows = nb_sims
            self.vec_returns = x
       
        elif data_type == 'real':
            
            ric = self.inputs.variable_name
            t = file_functions.load_timeseries(ric)
            
            self.data_table = t
            self.description = 'market data ' + ric
            self.nb_rows = t.shape[0]
            self.vec_returns = t['return_close'].values
            
            
    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vec_returns,bins=100)
        plt.title(self.description)
        plt.xlabel(self.plot_str())
        plt.show()
        
        
    def compute(self):
        self.mean = np.mean(self.vec_returns)
        self.std = np.std(self.vec_returns)
        self.skew = skew(self.vec_returns)
        self.kurtosis = kurtosis(self.vec_returns) # excess kurtosis
        self.jarque_bera_stat = self.nb_rows/6*(self.skew**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - chi2.cdf(self.jarque_bera_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
        self.var_95 = np.percentile(self.vec_returns,5)
        self.cvar_95 = np.mean(self.vec_returns[self.vec_returns <= self.var_95])
        self.percentile_25 = self.percentile(25)
        self.median = np.median(self.vec_returns)
        self.percentile_75 = self.percentile(75)

        
    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'percentile 25% ' + str(np.round(self.percentile_25,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | percentile 75% ' + str(np.round(self.percentile_75,nb_decimals))
        return plot_str
    
    
    def percentile(self, pct):
        percentile = np.percentile(self.vec_returns,pct)
        return percentile
    
    
    
class distribution_input():
    
    def __init__(self):
        self.data_type = None # simulation real custom
        self.variable_name = None # normal student exponential chi-square uniform VWS.CO
        self.degrees_freedom = None # only used in simulation + student and chi-square
        self.nb_sims = None # only in simulation
        
    
    
class capm_manager():

    def __init__(self,source_bench,source_ric,benchmark, ric):
        self.source_bench=source_bench
        self.source_ric=source_ric
        self.benchmark = benchmark # variable x
        self.ric = ric # variable y
        self.nb_decimals = 4
        self.nb_rows_ric = None
        self.nb_rows_bench = None
        self.data_table = None
        self.data_table1 = None
        self.df = None
        self.x = None
        self.y = None
        # self.data_table_benchmark = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = None
        self.correlation = None
        self.r_squared = None
        self.std_err = None
        self.predictor_linreg = None
        
        self.mean_bench = None
        self.std_bench = None
        self.skew_bench = None
        self.kurtosis_bench = None # excess kurtosis
        self.jarque_bera_stat_bench = None # under normality of self.vec_returns this distributes as chi-square with 2 degrees of freedom
        self.p_value_bench = None # equivalently jb < 6
        self.is_normal_bench = None
        self.sharpe_bench = None
        self.var_95_bench = None
        self.cvar_95_bench = None
        self.median_bench = None
        
        self.mean_ric = None
        self.std_ric = None
        self.skew_ric = None
        self.kurtosis_ric = None # excess kurtosis
        self.jarque_bera_stat_ric = None # under normality of self.vec_returns this distributes as chi-square with 2 degrees of freedom
        self.p_value_ric = None # equivalently jb < 6
        self.is_normal_ric = None
        self.sharpe_ric = None
        self.var_95_ric = None
        self.cvar_95_ric = None
        self.median_ric = None
        
        
        
    def __str__(self):
        return self.str_self()
    

    def str_self(self):
        nb_decimals = 4        
        str_self = 'Linear regression | ric ' + self.ric\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'correl (r-value) ' + str(self.correlation)\
            + ' | r-squared ' + str(self.r_squared) + '\n' #+ '\n'\
            # + '---'+self.benchmark+'---' + '\n'\
            # + 'mean ' + str(np.round(self.mean_bench,nb_decimals))\
            # + ' | std dev ' + str(np.round(self.std_bench,nb_decimals))\
            # + ' | skewness ' + str(np.round(self.skew_bench,nb_decimals))\
            # + ' | kurtosis ' + str(np.round(self.kurtosis_bench,nb_decimals)) + '\n'\
            # + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat_bench,nb_decimals))\
            # + ' | p-value ' + str(np.round(self.p_value_bench,nb_decimals))\
            # + ' | is normal ' + str(self.is_normal_bench) + '\n'\
            # + 'Sharpe annual ' + str(np.round(self.sharpe_bench,nb_decimals))\
            # + ' | VaR 95% ' + str(np.round(self.var_95_bench,nb_decimals))\
            # + ' | CVaR 95% ' + str(np.round(self.cvar_95_bench,nb_decimals)) + '\n'\
            # + ' | median ' + str(np.round(self.median_bench,nb_decimals))
        return str_self
    
   # def load_timeseries(self):
       # self.data_table = file_functions.load_synchronised_timeseries(ric_x=self.benchmark, ric_y=self.ric)    

    def load(self):
        
        if self.source_ric=='yahoo' and self.source_bench=='yahoo':
            self.data_table = file_functions.load_timeseries_ric_yahoo(self.ric)
            self.data_table1 = file_functions.load_timeseries_bench_yahoo(self.benchmark)
        elif self.source_ric=='investing' and self.source_bench=='investing':
            self.data_table = file_functions.load_timeseries_ric_investing(self.ric)
            self.data_table1 = file_functions.load_timeseries_bench_investing(self.benchmark)
        elif self.source_ric=='yahoo' and self.source_bench=='investing':
            self.data_table = file_functions.load_timeseries_ric_yahoo(self.ric)
            self.data_table1 = file_functions.load_timeseries_bench_investing(self.benchmark)
        elif self.source_ric=='investing' and self.source_bench=='yahoo':
            self.data_table = file_functions.load_timeseries_ric_investing(self.ric)
            self.data_table1 = file_functions.load_timeseries_bench_yahoo(self.benchmark)
                   
        # t1=self.data_table
        # t2=self.data_table1
        
        timestamp1=list(self.data_table['date'].values)
        timestamp2=list(self.data_table1['date'].values)
        timestamps=list(set(timestamp1) & set(timestamp2))
        
        t1_sync=self.data_table[self.data_table['date'].isin(timestamps)]
        t1_sync.sort_values(by='date', ascending=True)
        t1_sync=t1_sync.reset_index(drop=True)
        
        t2_sync=self.data_table1[self.data_table1['date'].isin(timestamps)]
        t2_sync.sort_values(by='date', ascending=True)
        t2_sync=t2_sync.reset_index(drop=True)
        
        
        # table of returns for ric and benchmark
        self.df=pd.DataFrame()
        self.df['date']=t1_sync['date']
        self.df['price1']=t1_sync['close']
        self.df['return1']=t1_sync['return_close']
        self.df['price2']=t2_sync['close']
        self.df['return2']=t2_sync['return_close']

        self.nb_rows_ric = self.df.shape[0]            
        self.nb_rows_bench = self.df.shape[0] 
        # compute vectors of returns
        self.y=self.df['return1'].values#ric
        self.x=self.df['return2'].values#bench
        
    def compute_jb_ric(self):
        self.mean_ric = np.mean(self.y)
        self.std_ric = np.std(self.y)
        self.skew_ric = skew(self.y)
        self.kurtosis_ric = kurtosis(self.y) # excess kurtosis
        self.jarque_bera_stat_ric = self.nb_rows_ric/6*(self.skew_ric**2 + 1/4*self.kurtosis_ric**2)
        self.p_value_ric = 1 - chi2.cdf(self.jarque_bera_stat_ric, df=2)
        self.is_normal_ric = (self.p_value_ric > 0.05) # equivalently jb < 6
        self.sharpe_ric = self.mean_ric / self.std_ric * np.sqrt(252) # annualised
        self.var_95_ric = np.percentile(self.y,5)
        self.cvar_95_ric = np.mean(self.y[self.y <= self.var_95_ric])
        # self.percentile_25_ric = self.percentile(25)
        self.median_ric = np.median(self.y)
        # self.percentile_75_ric = self.percentile(75)

    def compute_jb_bench(self):
        self.mean_bench = np.mean(self.x)
        self.std_bench = np.std(self.x)
        self.skew_bench = skew(self.x)
        self.kurtosis_bench = kurtosis(self.x) # excess kurtosis
        self.jarque_bera_stat_bench = self.nb_rows_bench/6*(self.skew_bench**2 + 1/4*self.kurtosis_bench**2)
        self.p_value_bench = 1 - chi2.cdf(self.jarque_bera_stat_bench, df=2)
        self.is_normal_bench = (self.p_value_bench > 0.05) # equivalently jb < 6
        self.sharpe_bench = self.mean_bench / self.std_bench * np.sqrt(252) # annualised
        self.var_95_bench = np.percentile(self.x,5)
        self.cvar_95_bench = np.mean(self.x[self.x <= self.var_95_bench])
        # self.percentile_25_bench = self.percentile(25)
        self.median_bench = np.median(self.x)
        # self.percentile_75_bench = self.percentile(75)
        
    def plot_str_ric(self):
        nb_decimals = 4
        plot_str_ric ='---'+self.ric+'---' + '\n'\
            + 'mean ' + str(np.round(self.mean_ric,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std_ric,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew_ric,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis_ric,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat_ric,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value_ric,nb_decimals))\
            + ' | is normal ' + str(self.is_normal_ric)\
            + ' | sample size ' + str(np.round(self.nb_rows_ric, nb_decimals)) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe_ric,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95_ric,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95_ric,nb_decimals))\
            + ' | median ' + str(np.round(self.median_ric,nb_decimals)) + '\n'
        return plot_str_ric
    
    def plot_str_bench(self):
        nb_decimals = 4
        plot_str_bench ='---'+self.benchmark+'---' + '\n'\
            + 'mean ' + str(np.round(self.mean_bench,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std_bench,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew_bench,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis_bench,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat_bench,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value_bench,nb_decimals))\
            + ' | is normal ' + str(self.is_normal_bench)\
            + ' | sample size ' + str(np.round(self.nb_rows_bench, nb_decimals)) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe_bench,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95_bench,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95_bench,nb_decimals))\
            + ' | median ' + str(np.round(self.median_bench,nb_decimals)) + '\n'
        return plot_str_bench
        
        # return x, y

    def plot_histogram_ric(self):
        plt.figure()
        plt.hist(self.y,bins=100)
        plt.title('Market Data '+self.ric)
        plt.xlabel(self.plot_str_ric())
        plt.show()
        
    def plot_histogram_bench(self):
        plt.figure()
        plt.hist(self.x,bins=100)
        plt.title('Market Data '+self.benchmark)
        plt.xlabel(self.plot_str_bench())
        plt.show()
        
        
    # def compute(self):
    #     self.mean = np.mean(self.vec_returns)
    #     self.std = np.std(self.vec_returns)
    #     self.skew = skew(self.vec_returns)
    #     self.kurtosis = kurtosis(self.vec_returns) # excess kurtosis
    #     self.jarque_bera_stat = self.nb_rows/6*(self.skew**2 + 1/4*self.kurtosis**2)
    #     self.p_value = 1 - chi2.cdf(self.jarque_bera_stat, df=2)
    #     self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
    #     self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
    #     self.var_95 = np.percentile(self.vec_returns,5)
    #     self.cvar_95 = np.mean(self.vec_returns[self.vec_returns <= self.var_95])
    #     self.percentile_25 = self.percentile(25)
    #     self.median = np.median(self.vec_returns)
    #     self.percentile_75 = self.percentile(75)

        
    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'percentile 25% ' + str(np.round(self.percentile_25,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | percentile 75% ' + str(np.round(self.percentile_75,nb_decimals))
        return plot_str
    

    
    def compute(self):
        # linear regression
        x = self.df['return2'].values
        y = self.df['return1'].values
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        self.alpha = np.round(intercept, self.nb_decimals)
        self.beta = np.round(slope, self.nb_decimals)
        self.p_value = np.round(p_value, self.nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.correlation = np.round(r_value, self.nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, self.nb_decimals) # pct of variance of y explained by x
        self.predictor_linreg = intercept + slope*x
        
        
    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        # plt.plot(self.data_table['date'],self.data_table['price2'], self.data_table['price1'])
        ax = plt.gca()
        ax1 = self.df['price2'].plot(kind='line', x=self.df['date'], ax=ax, grid=True,\
                                  color='blue', label=self.benchmark)
        ax2 = self.df['price1'].plot(kind='line', x=self.df['date'], ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.ric)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
    def plot_timeseries_normalised(self):
        p1=self.df['price1']
        p2=self.df['price2']
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices normalised at 100')
        plt.xlabel('Time')
        plt.ylabel('Normalised Prices')
        plt.plot(100*p1/p1[0], color='blue', label=self.ric)
        plt.plot(100*p2/p2[0], color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
    def plot_linear_regression(self):
        x = self.df['return2'].values
        y = self.df['return1'].values
        str_title = 'Scatterplot of returns' + '\n' + self.str_self()
        plt.figure()
        plt.title(str_title)
        plt.scatter(x,y)
        plt.plot(x, self.predictor_linreg, color='green')
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()



class hedge_manager():
    
    def __init__(self, inputs):
        self.inputs = inputs # hedge_inputs
        self.benchmark = inputs.benchmark # the market in CAPM, in general ^STOXX50E
        self.ric = inputs.ric # portfolio to hedge
        self.hedge_securities = inputs.hedge_securities # hedge universe
        self.nb_hedges = len(self.hedge_securities)
        self.portfolio_delta = inputs.delta_portfolio
        self.portfolio_beta = None
        self.portfolio_beta_usd = None
        self.betas = None
        self.optimal_hedge = None
        self.hedge_delta = None
        self.hedge_beta_usd = None
        self.regularisation = 0.0
    
    def load_betas(self):
        benchmark = self.benchmark
        ric = self.ric
        hedge_securities = self.hedge_securities
        portfolio_delta = self.portfolio_delta
        # compute beta for the portfolio
        capm = file_classes.capm_manager(benchmark, ric)
        capm.load_timeseries()
        capm.compute()
        portfolio_beta = capm.beta
        portfolio_beta_usd = portfolio_beta * portfolio_delta # mn USD
        # print input
        print('------')
        print('Input portfolio:')
        print('Delta mnUSD for ' + ric + ' is ' + str(portfolio_delta))
        print('Beta for ' + ric + ' vs ' + benchmark + ' is ' + str(portfolio_beta))
        print('Beta mnUSD for ' + ric + ' vs ' + benchmark + ' is ' + str(portfolio_beta_usd))
        # compute betas for the hedges
        shape = [len(hedge_securities)]
        betas = np.zeros(shape)
        counter = 0
        print('------')
        print('Input hedges:')
        for hedge_security in hedge_securities:
            capm = file_classes.capm_manager(benchmark, hedge_security)
            capm.load_timeseries()
            capm.compute()
            beta = capm.beta
            print('Beta for hedge[' + str(counter) + '] = ' + hedge_security + ' vs ' + benchmark + ' is ' + str(beta))
            betas[counter] = beta
            counter += 1
        
        self.portfolio_beta = portfolio_beta
        self.portfolio_beta_usd = portfolio_beta_usd
        self.betas = betas
        
        
    def compute(self, regularisation=0.0):
        # numerical solution
        dimensions = len(self.hedge_securities)
        x = np.zeros([dimensions,1])
        betas = self.betas
        optimal_result = minimize(fun=file_functions.cost_function_hedge, x0=x, args=(self.portfolio_delta, self.portfolio_beta_usd, betas, regularisation))
        self.optimal_hedge = optimal_result.x
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        self.regularisation = regularisation
        self.print_result('numerical')
        
        
    def compute_exact(self):
        # exact solution using matrix algebra
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('------')
            print('Cannot compute exact solution because dimensions = ' + str(dimensions) + ' =/= 2')
            return
        deltas = np.ones([dimensions])
        betas = self.betas
        targets = -np.array([[self.delta_portfolio],[self.beta_portfolio_usd]])
        mtx = np.transpose(np.column_stack((deltas,betas)))
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets)
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        self.print_result('exact')
        
    
    def print_result(self, algo_type):
        print('------')
        print('Optimisation result - ' + algo_type + ' solution')
        print('------')
        print('Delta portfolio: ' + str(self.portfolio_delta))
        print('Beta portfolio USD: ' + str(self.portfolio_beta_usd))
        print('------')
        print('Delta hedge: ' + str(self.hedge_delta))
        print('Beta hedge USD: ' + str(self.hedge_beta_usd))
        print('------')
        print('Optimal hedge:')
        print(self.optimal_hedge)
        

        
class hedge_input:
   
   def __init__(self):
       self.benchmark = None # the market in CAPM, in general ^STOXX50E
       self.ric = 'BBVA.MC' # portfolio to hedge
       self.hedge_securities =  ['^STOXX50E','^FCHI'] # hedge universe
       self.delta_portfolio = None # in mn USD, default 10   



class portfolio_manager:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.size = len(rics)
        self.notional = notional
        self.nb_decimals = 6
        self.scale = 252
        self.covariance_matrix = None
        self.correlation_matrix = None
        self.returns = None
        self.volatilities = None
        
        
    def compute_covariance_matrix(self, bool_print=True):
        # compute variance-covariance matrix by pairwise covariances
        rics = self.rics
        size = self.size
        mtx_covar = np.zeros([size,size])
        mtx_correl = np.zeros([size,size])
        vec_returns = np.zeros([size,1])
        vec_volatilities = np.zeros([size,1])
        returns = []
        for i in range(size):
            ric_x = rics[i]
            for j in range(i+1):
                ric_y = rics[j]
                t = file_functions.load_synchronised_timeseries(ric_x, ric_y)
                ret_x = t['return_x'].values
                ret_y = t['return_y'].values
                returns = [ret_x, ret_y]
                # covariances
                temp_mtx = np.cov(returns)
                temp_covar = self.scale*temp_mtx[0][1]
                temp_covar = np.round(temp_covar,self.nb_decimals)
                mtx_covar[i][j] = temp_covar
                mtx_covar[j][i] = temp_covar
                # correlations
                temp_mtx = np.corrcoef(returns)
                temp_correl = temp_mtx[0][1]
                temp_correl = np.round(temp_correl,self.nb_decimals)
                mtx_correl[i][j] = temp_correl
                mtx_correl[j][i] = temp_correl
                if j == 0:
                    temp_ret = ret_x
            # returns
            temp_mean = np.round(self.scale*np.mean(temp_ret), self.nb_decimals)
            vec_returns[i] = temp_mean
            # volatilities
            temp_volatility = np.round(np.sqrt(self.scale)*np.std(temp_ret), self.nb_decimals)
            vec_volatilities[i] = temp_volatility
        # compute eigenvalues and eigenvectors for symmetric matrices
        eigenvalues, eigenvectors = LA.eigh(mtx_covar)
        
        self.covariance_matrix = mtx_covar
        self.correlation_matrix = mtx_correl
        self.returns = vec_returns
        self.volatilities = vec_volatilities
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        if bool_print:
            print('----')
            print('Securities:')
            print(self.rics)
            print('----')
            print('Returns (annualised):')
            print(self.returns)
            print('----')
            print('Volatilities (annualised):')
            print(self.volatilities)
            print('----')
            print('Variance-covariance matrix (annualised):')
            print(self.covariance_matrix)
            print('----')
            print('Correlation matrix:')
            print(self.correlation_matrix)
            print('----')
            print('Eigenvalues:')
            print(self.eigenvalues)
            print('----')
            print('Eigenvectors:')
            print(self.eigenvectors)
            
            
    def compute_portfolio(self, portfolio_type='default', target_return=None):
        
        portfolio = portfolio_item(self.rics)
        
        if portfolio_type == 'min-variance':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[0] / sum(abs(self.eigenvalues))
            eigenvector = self.eigenvectors[:,0]
            if max(eigenvector) < 0:
                eigenvector = - eigenvector
            weights_normalised = eigenvector / sum(abs(eigenvector))
           
        elif portfolio_type == 'min-variance-l1':
            portfolio.type = portfolio_type   
            # initialise optimisation
            x = np.zeros([self.size,1])
            # initialise constraints
            cons = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
            # compute optimisation
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons)
            weights_normalised = res.x
            
        elif portfolio_type == 'min-variance-l2':
            portfolio.type = portfolio_type   
            # initialise optimisation
            x = np.zeros([self.size,1])
            # initialise constraints
            cons = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
            # compute optimisation
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons)
            weights_normalised = res.x / sum(abs(res.x))
           
        elif portfolio_type == 'long-only':
            portfolio.type = portfolio_type   
            # initialise optimisation
            x = np.zeros([self.size,1])
            # initialise constraints
            cons = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
            bnds = [(0, None) for i in range(self.size)]
            # compute optimisation
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons, bounds=bnds)
            weights_normalised = res.x
            
        elif portfolio_type == 'pca':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[-1] / sum(abs(self.eigenvalues))
            eigenvector = self.eigenvectors[:,-1]
            if max(eigenvector) < 0:
                eigenvector = - eigenvector
            weights_normalised = eigenvector / sum(abs(eigenvector))
            
        elif portfolio_type == 'default' or portfolio_type == 'equi-weight':
            portfolio.type = 'equi-weight'
            weights_normalised = 1 / self.size * np.ones([self.size])
            
        elif portfolio_type == 'volatility-weighted':
            portfolio.type = portfolio_type
            x = 1 / self.volatilities
            weights_normalised = 1 / np.sum(x) * x
            
        elif portfolio_type == 'markowitz':
            portfolio.type = portfolio_type
            if target_return == None:
                target_return = np.mean(self.returns)
            portfolio.target_return = target_return    
            # initialise optimisation
            x = np.zeros([self.size,1])
            # initialise constraints
            cons = [{"type": "eq", "fun": lambda x: np.transpose(self.returns).dot(x).item() - target_return},\
                    {"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
            bnds = [(0, None) for i in range(self.size)]
            # compute optimisation
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons, bounds=bnds)
            weights_normalised = res.x
        
        weights = self.notional * weights_normalised
        
        portfolio.weights = weights
        portfolio.notional = sum(abs(weights))
        portfolio.delta = sum(weights)
        portfolio.pnl_annual_usd = np.transpose(self.returns).dot(weights).item()
        portfolio.return_annual = np.transpose(self.returns).dot(weights_normalised).item()
        portfolio.volatility_annual = file_functions.compute_portfolio_volatility(weights_normalised, self.covariance_matrix)
        portfolio.volatility_annual_usd = file_functions.compute_portfolio_volatility(weights, self.covariance_matrix)
        portfolio.sharpe_annual = portfolio.return_annual / portfolio.volatility_annual
        
        return portfolio
 
            

class portfolio_item():
    
    def __init__(self, rics):
        self.rics = rics
        self.notional = 0.0
        self.type = ''
        self.weights = []
        self.delta = 0.0
        self.variance_explained = None
        self.pnl_annual_usd = None
        self.volatility_annual_usd = None
        self.target_return = None
        self.return_annual = None
        self.volatility_annual = None
        self.sharpe_annual = None


    def summary(self):
        print('-----')
        print('Portfolio type: ' + self.type)
        print('Rics:')
        print(self.rics)
        print('Weights:')
        print(self.weights)
        print('Notional (mnUSD): ' + str(self.notional))
        print('Delta (mnUSD): ' + str(self.delta))
        if not self.variance_explained == None:
            print('Variance explained: ' + str(self.variance_explained))
        if not self.pnl_annual_usd == None:
            print('Profit and loss annual (mn USD): ' + str(self.pnl_annual_usd))
        if not self.volatility_annual_usd == None:
            print('Volatility annual (mn USD): ' + str(self.volatility_annual_usd))
        if not self.target_return == None:
            print('Target return: ' + str(self.target_return))
        if not self.return_annual == None:
            print('Return annual: ' + str(self.return_annual))
        if not self.volatility_annual == None:
            print('Volatility annual: ' + str(self.volatility_annual))
        if not self.sharpe_annual == None:
            print('Sharpe ratio annual: ' + str(self.sharpe_annual))
            
            
class option_input:
    
    def __init__(self):
        self.price = None
        self.time = None
        self.volatility = None
        self.interest_rate = None
        self.maturity = None
        self.strike = None
        self.call_or_put = None
        
        
class montecarlo_item():
    
    def __init__(self, sim_prices, sim_payoffs, strike, call_or_put):
        self.number_simulations = len(sim_payoffs)
        self.sim_prices = sim_prices
        self.sim_payoffs = sim_payoffs
        self.call_or_put = call_or_put
        self.mean = np.mean(sim_payoffs)
        self.std = np.std(sim_payoffs)
        self.confidence_radius = 1.96*self.std/np.sqrt(self.number_simulations)
        self.confidence_interval =  self.mean + np.array([-1,1])*self.confidence_radius
        if call_or_put == 'call':
            self.proba_exercise = np.mean(sim_prices > strike)
        elif call_or_put == 'put':
            self.proba_exercise = np.mean(sim_prices < strike)
        self.proba_profit = np.mean(sim_payoffs > self.mean)
        
        
    def __str__(self):
        str_self = 'Monte Carlo simulation for option pricing | ' + self.call_or_put + '\n'\
            + 'number of simulations ' + str(self.number_simulations) + '\n'\
            + 'confidence radius ' + str(self.confidence_radius) + '\n'\
            + 'confidence interval ' + str(self.confidence_interval) + '\n'\
            + 'price ' + str(self.mean)  + '\n'\
            + 'probability of exercise ' + str(self.proba_exercise)  + '\n'\
            + 'probability of profit ' + str(self.proba_profit)  + '\n'\
                
        return str_self
    
    
    def plot_histogram(self):
        inputs_distribution = file_classes.distribution_input()
        dm = file_classes.distribution_manager(inputs_distribution)
        dm.description = 'Monte Carlo distribution | option price | ' + self.call_or_put
        dm.nb_rows = len(self.sim_payoffs)
        dm.vec_returns = self.sim_payoffs
        dm.compute() # compute returns and all different risk metrics
        dm.plot_histogram() # plot histogram
        print(dm) # write all data in console
        