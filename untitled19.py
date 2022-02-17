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
                                       'Stats', 'Charts and TA', 'News', 'Screener', 'Sec Filings',
                                       'Insider Transactions', 'Options'))


end=dt.datetime.now()
start=[end-dt.timedelta(days=365*10)]

apikey='d60d2f087ecf05f94a3b9b3df34310a9'
symbol= st.text_input("stock", '')

balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol)
income_statements=fmp.company_valuation.income_statement(apikey, symbol)
cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol)
            
balance_sheet=pd.DataFrame()
balance_sheet['index']=balance_sheets[0].keys()
balance_sheet=balance_sheet.set_index('index')
for i in range(0,len(balance_sheets)):
    balance_sheet[balance_sheets[i]['date']]=balance_sheets[i].values()
balance_sheet=balance_sheet[balance_sheet.columns[::-1]]

income_statement=pd.DataFrame()
income_statement['index']=income_statements[0].keys()
income_statement=income_statement.set_index('index')
for i in range(0,len(income_statements)):
    income_statement[income_statements[i]['date']]=income_statements[i].values()
income_statement=income_statement[income_statement.columns[::-1]]
            
cashflow_statement=pd.DataFrame()
cashflow_statement['index']=cashflow_statements[0].keys()
cashflow_statement=cashflow_statement.set_index('index')
for i in range(0,len(cashflow_statements)):
    cashflow_statement[cashflow_statements[i]['date']]=cashflow_statements[i].values()
cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]

dates=pd.DataFrame()
dates['index']=[0]
dates=dates.set_index('index')
for i in range(0,len(balance_sheets)):
    dates[balance_sheets[i]['date']]=str(balance_sheets[i]['date'])
dates=dates[dates.columns[::-1]]
dates=dates.transpose()  

if symbol != '':
    
    if option == 'Financials':
        
        data=st.sidebar.selectbox('Choose',('Balance Sheet','Income Statement','Cashflow Statement'))
        
        if data == 'Balance Sheet':
        
            st.header("Balance Sheet")
            st.markdown('Currency: '+str(balance_sheets[0]['reportedCurrency']))
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
            
            opt=st.selectbox('Histogram', ('totalCurrentAssets','totalNonCurrentAssets','totalCurrentLiabilities','totalNonCurrentLiabilities','totalEquity', 'netDebt',
            'cashAndCashEquivalents','shortTermInvestments','cashAndShortTermInvestments','netReceivables',
            'inventory','otherCurrentAssets','propertyPlantEquipmentNet','goodwill','intangibleAssets','goodwillAndIntangibleAssets',
            'longTermInvestments','taxAssets','otherNonCurrentAssets','totalNonCurrentAssets','otherAssets',
            'accountPayables','shortTermDebt','taxPayables','deferredRevenue','otherCurrentLiabilities',
            'longTermDebt','deferredRevenueNonCurrent','deferredTaxLiabilitiesNonCurrent','otherNonCurrentLiabilities',
            'otherLiabilities','capitalLeaseObligations','preferredStock','commonStock','retainedEarnings',
            'accumulatedOtherComprehensiveIncomeLoss','othertotalStockholdersEquity','totalStockholdersEquity','totalLiabilitiesAndStockholdersEquity',
            'minorityInterest'))
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            plt.title(opt)
            ax.bar(dates[0],balance_sheet.loc[opt])
            plt.yscale('log')
            st.pyplot(fig)
            
        if data == 'Income Statement':
        

            st.header("Income Statement")
            st.markdown('Currency: ' + income_statements[0]['reportedCurrency'])
            st.subheader("Revenues")
            revenues=income_statement.iloc[8:11,:]
            st.dataframe(revenues)
            st.subheader("Operating Expenses & Income")
            opex=income_statement.iloc[12:19,:]
            opex.loc['operatingIncome']=income_statement.loc['operatingIncome']
            st.dataframe(opex)
            st.subheader("EBITDA")
            ebitda=income_statement.iloc[20:23,:]
            st.dataframe(ebitda)
            st.subheader("EBT")
            ebt=income_statement.iloc[26:28,:]
            st.dataframe(ebt)
            st.subheader("Net Income")
            net_inc=income_statement.iloc[29:31,:]
            st.dataframe(net_inc)
            
            opt=st.selectbox('Histogram', ('revenue', 'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses',
                                           'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses',
                                           'otherExpenses', 'operatingExpenses', 'operatingIncome', 'costAndExpenses',
                                           'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'totalOtherIncomeExpensesNet',
                                           'incomeBeforeTax', 'incomeTaxExpense', 'netIncome', 'eps', 'epsdiluted', 'weightedAverageShsOut','weightedAverageShsOutDil' ))
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            plt.title(opt)
            ax.bar(dates[0],income_statement.loc[opt])
            plt.yscale('log')
            st.pyplot(fig)
            
            

        
        if data == 'Cashflow Statement':
        

            
            st.header("Cashflow Statement")    
            st.dataframe(cashflow_statement)
            
    if option == 'Valuation':
        opt=st.selectbox('Options', ('Main','Liquidity & Solvency','Profitability & Efficiency','Valuation'))
        ls=pd.DataFrame()
        ls['index']=dates
        ls=ls.set_index('index')
        
        
        data=pd.DataFrame()
        data['index']=dates
        data=data.set_index('index')
        
        #Data
        dex=(income_statement.loc['costOfRevenue']+income_statement.loc['operatingExpenses']-income_statement.loc['depreciationAndAmortization'])/365
        # avg_inv=
        
        #Activity Ratios
        # ls['Inventary turnover']=
        
        # Liquidity 
        ls['Current ratio']=balance_sheet.loc['totalCurrentAssets']/balance_sheet.loc['totalCurrentLiabilities']
        ls['Quick ratio']=(balance_sheet.loc['cashAndShortTermInvestments']+balance_sheet.loc['netReceivables'])/balance_sheet.loc['totalCurrentLiabilities']
        ls['Cash ratio']=balance_sheet.loc['cashAndShortTermInvestments']/balance_sheet.loc['totalCurrentLiabilities']
        ls['Defensive interval']=(balance_sheet.loc['cashAndShortTermInvestments']+balance_sheet.loc['netReceivables'])/dex
        #Solvency
        ls['Debt to assets']=balance_sheet.loc['totalLiabilities']/balance_sheet.loc['totalAssets']
        ls['Debt to capital']=balance_sheet.loc['totalLiabilities']/(balance_sheet.loc['totalStockholdersEquity']+balance_sheet.loc['totalLiabilities'])
        ls['Debt to equity']=balance_sheet.loc['totalLiabilities']/balance_sheet.loc['totalStockholdersEquity']
        ls['Leverage ratio']=balance_sheet.loc['totalAssets']/balance_sheet.loc['totalStockholdersEquity']
        ls['FFO to Debt']=(income_statement.loc['netIncome']+income_statement.loc['depreciationAndAmortization']-income_statement.loc['interestIncome'])/balance_sheet.loc['totalLiabilities']
        
        
        #Coverage
        ls['Interest coverage']=(income_statement.loc['ebitda']-income_statement.loc['depreciationAndAmortization'])/(income_statement.loc['interestExpense']-income_statement.loc['interestIncome'])
        ls['EBIT interest coverage']=(income_statement.loc['ebitda']-income_statement.loc['depreciationAndAmortization'])/income_statement.loc['interestExpense']
        ls['EBITDA interest coverage']=income_statement.loc['ebitda']/income_statement.loc['interestExpense']
        ls['FFO interest coverage']=(income_statement.loc['netIncome']+income_statement.loc['depreciationAndAmortization']-income_statement.loc['interestIncome'])/income_statement.loc['interestExpense']
        
        
        #Profitability(margenes-returns-)
        ls['Gross profit margin']=income_statement.loc['grossProfitRatio']
        ls['Operating Margin']=income_statement.loc['operatingIncomeRatio']
        ls['EBT Margin']=income_statement.loc['incomeBeforeTaxRatio']
        ls['Net profit margin']=income_statement.loc['netIncomeRatio']
        
        ls['Operating ROA']=income_statement.loc['operatingIncome']/balance_sheet.loc['totalAssets']
        ls['ROA']=income_statement.loc['netIncome']/balance_sheet.loc['totalAssets']#average
        ls['ROTA']=(income_statement.loc['ebitda']-income_statement.loc['depreciationAndAmortization'])/balance_sheet.loc['totalAssets']
        ls['ROE']=income_statement.loc['netIncome']/balance_sheet.loc['totalEquity']
        
        #Valuation
        
        

        
        
        
    
        if opt == 'Main':
            ratioss=fmp.company_valuation.financial_ratios_ttm(apikey, symbol)
            ratios=pd.DataFrame()
            ratios['index']=ratioss[0].keys()
            ratios=ratios.set_index('index')
            
            ratios['Value']=ratioss[0].values()
            ratios=ratios[ratios.columns[::-1]]
            
            ratios=ratios.astype(str)
            
            st.subheader("Main Ratios")    
            st.dataframe(ratios)
        if opt == 'Liquidity & Solvency':
            l=ls.transpose()            
            li=l.iloc[0:4,:]
            lit=l.iloc[0:3,:].transpose()
            so=l.iloc[4:9,:]
            st.subheader('Liquidity')
            st.dataframe(li)
            st.subheader('Solvency')
            st.dataframe(so)
            
            pl=st.selectbox('Plot', ('Liquidity', 'Solvency'))
            if pl == 'Liquidity':
                fig=plt.figure(figsize=(12,5))
                plt.title('Liquidity ratios')
                plt.xlabel('Time')
                # plt.plot(self.data_table['date'],self.data_table['price2'], self.data_table['price1'])
                ax = plt.gca()
                for elements in lit:
                    elements=lit[elements].plot(kind='line', ax=ax, grid=True, label=elements)
                    elements.legend(loc=2)
                plt.yscale('log')    

                st.pyplot(fig)

            
            na=st.multiselect('Plot', ['','Current ratio','Quick ratio','Cash ratio','Defensive interval',
                                      'Debt to assets','Debt to capital','Debt to equity','Leverage ratio','FFO to Debt' ])
        if opt=='Profitability & Efficiency':
            o=ls.transpose()
            
            i=o.iloc[9:12]
            n=o.iloc[16:21,:]
            l=o.iloc[13:16,:]
            st.subheader('Interest coverage')

            st.dataframe(i)
            st.subheader('Efficiency')
            st.dataframe(n)
            st.subheader('Profitability')
            st.dataframe(l)

            
    if option == 'Price Chart':
        st.subheader('Intraday')
        prices=pd.DataFrame()
        pricess=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo').json()['Time Series (5min)']

    if option == 'Stats':
        
        run = 1

    if option == 'Social Sentiment':
        lol=st.selectbox('Options', ('Stocktwits','Twitter', 'Reddit', 'Google Trends'))
        if lol == 'Stocktwits':
            nft='NFT'
        
        
        
        
        