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
import plotly.graph_objects as go


# import file_classes
# importlib.reload(file_classes)
# import file_functions
# importlib.reload(file_functions)
# import untitled11
# importlib.reload(untitled11)
symbol= st.text_input("stock", '')
url= f'https://financialmodelingprep.com/image-stock/{symbol}.png'
st.image(url)
option=st.sidebar.selectbox("Options",('Home', 'Financials', 'Valuation', 'News',
                                       'Stats', 'Technical analysis', 'Screener', 'Sec Filings',
                                       'Insider Transactions', 'Options', 'Social Sentiment'))


end=dt.datetime.now()
start=[end-dt.timedelta(days=365*10)]

apikey='d60d2f087ecf05f94a3b9b3df34310a9'



if symbol != '':
    
    if option == 'Home':
        o=fmp.company_profile(apikey, symbol)

        
        st.header(str(o[0]['companyName'])+' | '+str(o[0]['exchange'])+' | '+str(o[0]['symbol'])+
                  ' | ISIN:'+str(o[0]['isin'])+' | CIK:'+str(o[0]['cik']))
        st.metric('Price',str(o[0]['price'])+' '+o[0]['currency'], str(np.round(o[0]['changes'],4))+'  |  '+str(np.round(np.round(o[0]['changes'],4)/(o[0]['price']-np.round(o[0]['changes'],4))*100,4))+'%')
        st.subheader('Sector and industry: '+str(o[0]['sector'])+' | '+str(o[0]['industry']))       
        st.subheader('Company profile')
        st.markdown(str(o[0]['description']))
        st.subheader('Website')
        st.markdown(o[0]['website'])
        # st.subheader('')
        # st.markdown(str(o[0]['']))
        # st.subheader('')
        # st.markdown(str(o[0]['']))
        # st.subheader('')
        # st.markdown(str(o[0]['']))
    if option == 'Financials':
        
        data=st.sidebar.selectbox('Choose',('Balance Sheet','Income Statement','Cashflow Statement'))
        
        if data == 'Balance Sheet':
            balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol, 'quarter', 30)                
            
            balance_sheet=pd.DataFrame(balance_sheets)
            balance_sheet=balance_sheet.set_index('date')
            balance_sheet=balance_sheet.transpose()
            balance_sheet=balance_sheet[balance_sheet.columns[::-1]]
            dates=pd.DataFrame(balance_sheet.columns)

            st.header("Balance Sheet")
            st.markdown('Currency: '+str(balance_sheets[0]['reportedCurrency']))
            st.subheader("Current Assets")
            current_ass = balance_sheet.iloc[7:14,:]
            st.dataframe(current_ass)            
            st.subheader("Non Current Assets")            
            non_curr_ass = balance_sheet.iloc[15:24,:]
            st.dataframe(non_curr_ass)             
            st.subheader("Current Liabilities")            
            curr_liab = balance_sheet.iloc[25:30,:]
            st.dataframe(curr_liab)             
            st.subheader("Non Current Liabilities")            
            non_curr_liab = balance_sheet.iloc[31:38,:]
            st.dataframe(non_curr_liab)             
            st.subheader("Equity")            
            equity = balance_sheet.iloc[38:51,:]
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
            ax.bar(dates['date'],balance_sheet.loc[opt])
            # plt.yscale('log')
            st.pyplot(fig)
            
        if data == 'Income Statement':
            income_statements=fmp.company_valuation.income_statement(apikey, symbol, 'quarter', 30)

            income_statement=pd.DataFrame(income_statements)
            income_statement=income_statement.set_index('date')
            income_statement=income_statement.transpose()
            income_statement=income_statement[income_statement.columns[::-1]]
            dates=pd.DataFrame(income_statement.columns)
            
            

            st.header("Income Statement")
            st.markdown('Currency: ' + income_statements[0]['reportedCurrency'])
            st.subheader("Revenues")
            revenues=income_statement.iloc[7:10,:]
            st.dataframe(revenues)
            st.subheader("Operating Expenses & Income")
            opex=income_statement.iloc[11:18,:]
            opex.loc['operatingIncome']=income_statement.loc['operatingIncome']
            st.dataframe(opex)
            st.subheader("EBITDA")
            ebitda=income_statement.iloc[19:22,:]
            st.dataframe(ebitda)
            st.subheader("EBT")
            ebt=income_statement.iloc[25:27,:]
            st.dataframe(ebt)
            st.subheader("Net Income")
            net_inc=income_statement.iloc[28:30,:]
            st.dataframe(net_inc)
            
            opt=st.selectbox('Histogram', ('revenue', 'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses',
                                           'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses',
                                           'otherExpenses', 'operatingExpenses', 'operatingIncome', 'costAndExpenses',
                                           'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'totalOtherIncomeExpensesNet',
                                           'incomeBeforeTax', 'incomeTaxExpense', 'netIncome', 'eps', 'epsdiluted', 'weightedAverageShsOut','weightedAverageShsOutDil' ))
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            plt.title(opt)
            ax.bar(dates['date'],income_statement.loc[opt])
            plt.yscale('log')
            st.pyplot(fig)
            

        
        if data == 'Cashflow Statement':
            cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol, 'quarter', 30)

            cashflow_statement=pd.DataFrame(cashflow_statements)
            cashflow_statement=cashflow_statement.set_index('date')
            cashflow_statement=cashflow_statement.transpose()
            cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]
            dates=pd.DataFrame(cashflow_statement.columns)

            
            st.header("Cashflow Statement")    
            st.dataframe(cashflow_statement)
            
            
            # opt=st.selectbox('Histogram', (
            
    if option == 'Valuation':

        
        balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol, 'quarter')
                
        balance_sheet=pd.DataFrame(balance_sheets)
        balance_sheet=balance_sheet.set_index('date')
        balance_sheet=balance_sheet.transpose()
        balance_sheet=balance_sheet[balance_sheet.columns[::-1]]
        dates=pd.DataFrame(balance_sheet.columns)

        
        income_statements=fmp.company_valuation.income_statement(apikey, symbol, 'quarter')

        income_statement=pd.DataFrame(income_statements)
        income_statement=income_statement.set_index('date')
        income_statement=income_statement.transpose()
        income_statement=income_statement[income_statement.columns[::-1]]

        cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol, 'quarter')

        cashflow_statement=pd.DataFrame(cashflow_statements)
        cashflow_statement=cashflow_statement.set_index('date')
        cashflow_statement=cashflow_statement.transpose()
        cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]
        
        opt=st.selectbox('Options', ('Main','Liquidity & Solvency','Profitability & Efficiency','Valuation'))
        ls=pd.DataFrame()
        ls['index']=dates
        ls=ls.set_index('index')
        ps=pd.DataFrame()        
        
        data=pd.DataFrame()
        data['index']=dates
        data=data.set_index('index')
        
        #Data
        dex=(income_statement.loc['costOfRevenue']+income_statement.loc['operatingExpenses']-income_statement.loc['depreciationAndAmortization'])/365
        price=fmp.stock_time_series.quote_short(apikey, symbol)
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
        
        
        rats=fmp.financial_ratios(apikey, symbol, 'quarter', 200)
        
        
        
    
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
            sot=so.transpose()
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
                # plt.yscale('log')    

                st.pyplot(fig)
            if pl == 'Solvency':
                fig=plt.figure(figsize=(12,5))
                plt.title('Solvency ratios')
                plt.xlabel('Time')
                # plt.plot(self.data_table['date'],self.data_table['price2'], self.data_table['price1'])
                ax = plt.gca()
                for elements in sot:
                    elements=sot[elements].plot(kind='line', ax=ax, grid=True, label=elements)
                    elements.legend(loc=2)
                # plt.yscale('log')    

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

            
    if option == 'Technical analysis':
        st.subheader('Technical analysis')
        prices=pd.DataFrame(fmp.historical_price_full(apikey, symbol))
        # pricess=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo').json()['Time Series (5min)']
        
        # i=st.selectbox('Intraday'('4H', '1D', '1W', '1M'))
        
        sma50=fmp.technical_indicators(apikey, symbol,50, 'sma')
        sma100=fmp.technical_indicators(apikey, symbol,100, 'sma')
        sma200=fmp.technical_indicators(apikey, symbol,200, 'sma')
        
        fig = go.Figure(data=go.Candlestick(x=prices.columns,
                        open=prices.loc['open'],
                        high=prices.loc['high'],
                        low=prices.loc['low'],
                        close=prices.loc['close']))
        
        st.plotly_chart(fig)
        
        
        
    if option == 'Stats':
        
        run = 1

    if option == 'Social Sentiment':
        lol=st.selectbox('Options', ('Stocktwits','Twitter', 'Reddit', 'Google Trends'))
        if lol == 'Stocktwits':
            nft='NFT'
            
    if option== 'News':
        select=st.selectbox('Options', ('News and articles','Press releases'))
        if select == 'News and articles':
            source=st.selectbox('Source', ('FMP','Marketaux'))
            if source == 'FMP':
              news=fmp.stock_news(apikey, symbol, 20)
              st.image(news[0]['image'])
              for i in range(0,len(news)):
                  st.subheader('*'+news[i]['site']+'*'+' | '+news[i]['title'])
                  st.markdown(news[i]['text'])
                  st.markdown(news[i]['publishedDate']+' | '+news[i]['url'])
            if source == 'Marketaux':
              r=requests.get(f'https://api.marketaux.com/v1/news/all?symbols={symbol}&filter_entities=true&language=en&api_token=1ZG29ExpDNcP5ka065e0h4rxpasuJeWqwu4Qgj9c').json()
              st.image(r['data'][0]['image_url'])  
              for i in range(0,len(r['data'])):
                  st.header(r['data'][i]['title'])
                  st.markdown(r['data'][i]['description'])
                  st.markdown(r['data'][i]['url'])  
        
        
        
        
