# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 22:25:37 2022

@author: diegu
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import fmpsdk as fmp
import importlib
import datetime as dt
import matplotlib.pyplot as plt
import plotly as px


# import file_classes
# importlib.reload(file_classes)
# import file_functions
# importlib.reload(file_functions)
# import untitled11
# importlib.reload(untitled11)


option=st.sidebar.selectbox("Options",('Home', 'Financials', 'Valuation', 'Social Sentiment', 'lol',
                                       'Ratios', 'Stats', 'Charts and TA', 'News', 'Screener', 'Sec Filings', 'Insider Transactions', 'Options'))


end=dt.datetime.now()
start=[end-dt.timedelta(days=365*10)]

apikey='d60d2f087ecf05f94a3b9b3df34310a9'
symbol= st.text_input("stock", '')
if symbol != '':
    if option == 'Financials':
        data=st.sidebar.selectbox('Choose',('Balance Sheet','Income Statement','Cashflow Statement'))
        if data == 'Balance Sheet':
        
            balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol)
            balance_sheet=pd.DataFrame()
            balance_sheet['index']=balance_sheets[0].keys()
            balance_sheet=balance_sheet.set_index('index')
            
            for i in range(0,len(balance_sheets)):
            
                balance_sheet[balance_sheets[i]['date']]=balance_sheets[i].values()
            balance_sheet=balance_sheet[balance_sheet.columns[::-1]]
            # balance_sheet=balance_sheet.astype(str)
            st.header("Balance Sheet")
            st.subheader("Current Assets")
            current_ass = balance_sheet.iloc[8:15,:]
            st.dataframe(current_ass)            
            st.subheader("Non Current Assets")            
            non_curr_ass = balance_sheet.iloc[16:25,:]
            st.dataframe(non_curr_ass)             
            st.subheader("Current Liabilities")            
            curr_liab = balance_sheet.iloc[26:31,:]
            st.dataframe(curr_liab)             
            st.subheader("Non Current Liabilities")            
            non_curr_liab = balance_sheet.iloc[32:39,:]
            st.dataframe(non_curr_liab)             
            st.subheader("Equity")            
            equity = balance_sheet.iloc[39:52,:]
            st.dataframe(equity)      
            opt=st.selectbox('Histogram', ('Current Assets','Non Current Assets','Current Liabilities','Non Current Liabilities','Equity', 'Net Debt',
            'cashAndCashEquivalents','shortTermInvestments','cashAndShortTermInvestments','netReceivables',
            'inventory','otherCurrentAssets','propertyPlantEquipmentNet','goodwill','intangibleAssets','goodwillAndIntangibleAssets',
            'longTermInvestments','taxAssets','otherNonCurrentAssets','totalNonCurrentAssets','otherAssets',
            'accountPayables','shortTermDebt','taxPayables','deferredRevenue','otherCurrentLiabilities',
            'longTermDebt','deferredRevenueNonCurrent','deferredTaxLiabilitiesNonCurrent','otherNonCurrentLiabilities',
            'otherLiabilities','capitalLeaseObligations','preferredStock','commonStock','retainedEarnings',
            'accumulatedOtherComprehensiveIncomeLoss','othertotalStockholdersEquity','totalStockholdersEquity','totalLiabilitiesAndStockholdersEquity',
            'minorityInterest'))
            df_statement=pd.DataFrame()
            for i in range(0,len(balance_sheets)):
                df_statement[balance_sheets[i]['date']]=balance_sheets[i][opt]
                
            fig=px.plot(df_statement,kind='histogram')
            st.plotly_chart(fig)
            
            
            # if opt ==  'Current Assets':
            #     fig=px.bar(balance_sheet.iloc[14])
            #     st.image(plt.show())
            
            # balplot=px.data.gapminder().query(opt)
            # fig=px.bar(balplot)
            # st.plotly_chart(fig)
 
            
        if data == 'Income Statement':
        
            income_statements=fmp.company_valuation.income_statement(apikey, symbol)
            income_statement=pd.DataFrame()
            income_statement['index']=income_statements[0].keys()
            income_statement=income_statement.set_index('index')
            for i in range(0,len(income_statements)):
                
                income_statement[income_statements[i]['date']]=income_statements[i].values()
            income_statement=income_statement[income_statement.columns[::-1]]
            income_statement=income_statement.astype(str)
            st.header("Income Statement")
            st.dataframe(income_statement)
        
        if data == 'Cashflow Statement':
        
            cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol)
            cashflow_statement=pd.DataFrame()
            cashflow_statement['index']=cashflow_statements[0].keys()
            cashflow_statement=cashflow_statement.set_index('index')
            for i in range(0,len(cashflow_statements)):
            
                cashflow_statement[cashflow_statements[i]['date']]=cashflow_statements[i].values()
            cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]
            
            cashflow_statement=cashflow_statement.astype(str)
        
            
        
            
            st.header("Cashflow Statement")    
            st.dataframe(cashflow_statement)
            
    if option == 'Ratios':
    
        ratioss=fmp.company_valuation.financial_ratios_ttm(apikey, symbol)
        ratios=pd.DataFrame()
        ratios['index']=ratioss[0].keys()
        ratios=ratios.set_index('index')
        
        ratios['Value']=ratioss[0].values()
        ratios=ratios[ratios.columns[::-1]]
        
        ratios=ratios.astype(str)
        
        st.subheader("Main Ratios")    
        st.dataframe(ratios)

    if option == 'Price Chart':
        st.subheader('Intraday')
        prices=pd.DataFrame()
        pricess=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo').json()['Time Series (5min)']

    if option == 'Stats':
        run = 1
