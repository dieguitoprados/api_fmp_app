# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 22:25:37 2022

@author: diegu
"""
import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import requests
import fmpsdk as fmp
import importlib
import datetime as dt
import matplotlib.pyplot as plt
import plotly as pt
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as scipy
apikey='d60d2f087ecf05f94a3b9b3df34310a9'

main=st. selectbox('APPS', ('Home', 'Stocks', 'ETFs & Indices','Cryptocurrencies','Forex', 'Fixed income', 'Commodities'))
if main == 'Home':
    ch=st.selectbox('Quick options:', ('Wire news', 'Calendars','Main indices','Hot stocks', 'Commodities'  ))
    
    if ch=='Wire news':
        r=requests.get('https://api.nytimes.com/svc/news/v3/content/all/all.json?api-key=XJ0LXsOASAnkltrGDXBpxFnOxlxpBsg9').json()
        
        
        for i in range(0,len(r['results'])):
            st.header(r['results'][i]['title'])
            st.subheader(r['results'][i]['first_published_date'])
            st.markdown(r['results'][i]['url'])
            
        
    if ch == 'Calendars':
        d = st.date_input('Date', dt.datetime.now() )
        ty=st.selectbox('Calendars', ('Economic', 'Earnings', 'IPO', 'Dividends'))
        if ty == 'Economic':
            ceco=fmp.calendar.economic_calendar(apikey, )
            for i in range (0, len(ceco)):
                while ceco[i]['date'] == d:
                    st.header(str(d))
                    
        
        ce=fmp.calendar.earning_calendar(apikey)
        
        
    if ch== 'Main indices':
        
        ind=fmp.indexes(apikey)
        st.header('Americas')
        st.subheader('USA')
        for i in [58, 59]:
            st.metric(str(ind[i]['name']),str (ind[i]['price']), str(ind[i]['change'])+ ' | ' +str(ind[i]['changesPercentage'])+'%')
        # col[1].markdown(ind[][])
        
        
        
    if ch== 'Hot stocks':
        col1, col2, col3=st.columns((3))
        
        i=fmp.indexes(apikey)
        a=fmp.stock_market.actives(apikey)
        g=fmp.stock_market.gainers(apikey)
        l=fmp.stock_market.losers(apikey)
        p=fmp.stock_market.sectors_performance(apikey)
        
        col1.subheader('Gainers')
        for i in range (0,9):
            col1.markdown(g[i]['ticker']+' || '+g[i]['companyName']+' || '+'📈'+g[i]['changesPercentage']+'%')
        col2.subheader('Loosers')
        for i in range (0,9):
            col2.markdown(l[i]['ticker']+' || '+l[i]['companyName']+' || '+'📉'+l[i]['changesPercentage']+'%')
        ...
        with col3:
        
            st.subheader('Active')
            for i in range (0,9):
                st.markdown((a[i]['ticker']+' || '+a[i]['companyName']+' || '+a[i]['changesPercentage']+'%'))
            
    # if ch == 'Commodities':
    
if main == 'Stocks':



    symbol= st.text_input("stock", '')
    url= f'https://financialmodelingprep.com/image-stock/{symbol}.png'
    st.image(url)
    option=st.sidebar.selectbox("Options",('Home', 'Financials', 'Horizontal', 'Valuation', 'News',
                                           'Stats', 'Technical analysis', 'Screener', 'SEC Filings',
                                           'Insider Transactions', 'Options', 'Social Sentiment'))
    
    
    end=dt.datetime.now()
    start=[end-dt.timedelta(days=365*10)]

    prices=fmp.historical_chart(apikey, symbol, '4hour')
    p=pd.DataFrame(prices)
    p=p.set_index(p['date'])
    # fig = px.line(p, x='date', y='close', title='Price of '+symbol+' || '+"price in "+fmp.balance_sheet_statement(apikey, symbol)[0]['reportedCurrency'])
    
    # st.plotly_chart(fig)
    
    if symbol != '':
        
        if option == 'Home':
            o=fmp.company_profile(apikey, symbol)
    
            
            st.header(str(o[0]['companyName'])+' | '+str(o[0]['exchange'])+' | '+str(o[0]['symbol'])+
                      ' | ISIN:'+str(o[0]['isin'])+' | CIK:'+str(o[0]['cik']))
            st.metric('Price',str(o[0]['price'])+' '+o[0]['currency'], str(np.round(o[0]['changes'],4))+'  |  '+str(np.round(np.round(o[0]['changes'],4)/(o[0]['price']-np.round(o[0]['changes'],4))*100,4))+'%')
            st.markdown('Sector and industry: '+str(o[0]['sector'])+' | '+str(o[0]['industry'])) 
            
            
            
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
        if option== 'Horizontal':
            opti=st.selectbox('Choose:', ('Main', 'Balance Sheet', 'Income Statement', 'Cash Flow'))
            
            if opti=='Main':
            
                rating=fmp.historical_rating(apikey, symbol, 15)
                r=pd.DataFrame(rating)
                r=r.set_index('date')
                r=r[r.columns[2:]]
                rc=r.columns
                r=r.transpose()
                r=r[r.columns[::-1]]
                st.dataframe(r)
                opt=st.selectbox('Plot', (rc))
                st.bar_chart(r.loc[opt])
                
            if opti=='Balance Sheet':
            
                bg=fmp.balance_sheet_statement_growth(apikey, symbol, 15)
                bgg=pd.DataFrame(bg)
                bgg=bgg.set_index('date')
                bgg=bgg[bgg.columns[2:]]
                bgc=bgg.columns
                bgg=bgg.transpose()
                bgg=bgg[bgg.columns[::-1]]
                st.dataframe(bgg)
                opt=st.selectbox('Plot', bgc)
                st.bar_chart(bgg.loc[opt])
                
            if opti=='Income Statement':
            
                bg=fmp.income_statement_growth(apikey, symbol, 15)
                bgg=pd.DataFrame(bg)
                bgg=bgg.set_index('date')
                bgg=bgg[bgg.columns[2:]]
                bgc=bgg.columns
                bgg=bgg.transpose()
                bgg=bgg[bgg.columns[::-1]]
                st.dataframe(bgg)
                opt=st.selectbox('Plot', bgc)
                st.bar_chart(bgg.loc[opt])
                
            if opti=='Balance Sheet':
            
                bg=fmp.cash_flow_statement(apikey, symbol, 15)
                bgg=pd.DataFrame(bg)
                bgg=bgg.set_index('date')
                bgg=bgg[bgg.columns[2:]]
                bgc=bgg.columns
                bgg=bgg.transpose()
                bgg=bgg[bgg.columns[::-1]]
                st.dataframe(bgg)
                opt=st.selectbox('Plot', bgc)
                st.bar_chart(bgg.loc[opt])
            
            
            
        if option == 'Financials':
            
            data=st.sidebar.selectbox('Choose',('Balance Sheet','Income Statement','Cashflow Statement'))
            
            if data == 'Balance Sheet':
                
                time=st.radio('Time Interval:',( 'quarter','annual'))
                
                balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol, time, 30)                
                # bsar=fmp.balance_sheet_statement_as_reported(apikey, symbol, 'quarter', 30)

                # bb=pd.DataFrame(bsar)
                # bb=bb.transpose()

                # ii=pd.DataFrame(fmp.income_statement_as_reported(apikey, symbol, 'quarter', 30))
                # ii=ii.transpose()

                balance_sheet=pd.DataFrame(balance_sheets)
                balance_sheet=balance_sheet.set_index('date')
                bsc=balance_sheet[balance_sheet.columns[7:]]
                bsc=bsc.columns
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
                
                opt=st.selectbox('Histogram', (bsc))
                
                chtype=st.radio('Chart type:', ('st.chart', 'Pyplot'))
                
                if chtype == 'st.chart':
                
                    st.bar_chart(balance_sheet.loc[opt])
                
                if chtype== 'Pyplot':
                    
                    fig = plt.figure()
                    ax = fig.add_axes([0,0,1,1])
                    plt.title(opt)
                    ax.bar(dates['date'],balance_sheet.loc[opt])
                    # plt.yscale('log')
                    st.pyplot(fig)
                    
                if data == 'Income Statement':
                    income_statements=fmp.company_valuation.income_statement(apikey, symbol, 'quarter', 30)
        
                    income_statement=pd.DataFrame(income_statements)
                    income_statement=income_statement.set_index(income_statement['calendarYear']+' '+income_statement['period'])
                    income_statement=income_statement.transpose()
                    income_statement=income_statement[income_statement.columns[::-1]]
                    dates=pd.DataFrame(income_statement.columns)
                    # dq = [dates[i] for i in range(len(dates)) if i % 2 != 0]                
                    
        
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
                    # plt.xticks(dq)
                    # plt.yscale('log')
                    st.pyplot(fig)
                
    
            
            if data == 'Cashflow Statement':
                cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol, 'quarter', 30)
    
                cashflow_statement=pd.DataFrame(cashflow_statements)
                cashflow_statement=cashflow_statement.set_index('date')
                cashflow_statement=cashflow_statement.transpose()
                cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]
                dates=pd.DataFrame(cashflow_statement.columns)
    
                
                st.header("Cashflow Statement")
                st.markdown('Currency: ' + cashflow_statements[0]['reportedCurrency'])
    
                st.dataframe(cashflow_statement.iloc[7:,:])
                
                
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
            
            opt=st.selectbox('Options', ('Compare','Liquidity & Solvency','Profitability & Efficiency','Valuation', 'Discounted Cash Flow'))
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
            
            
            rats=fmp.financial_ratios(apikey, symbol, 'quarter', 20)
            rf=pd.DataFrame(rats)
            rf=rf.set_index('date')
            rf=rf[rf.columns[2:]]
            rc=rf.columns
            rf=rf.transpose()
            rf=rf[rf.columns[::-1]]
            
            
        
            if opt == 'Compare':
                
                
                fmp.discounted_cash_flow(apikey, symbol)
                fmp.enterprise_values(apikey, symbol, 'quarter', 20)
                lp=fmp.historical_daily_discounted_cash_flow(apikey, symbol, 300)
                st.subheader("Main Ratios")    
                st.dataframe(rf)
                ox=st.selectbox('Plot:', rf.index)
                
                fig=plt.figure()
                plt.figure(figsize=(12,5))
                plt.title('Time series of'+ox)
                plt.xlabel('Time')
                plt.ylabel(ox)
                # plt.plot(self.data_table['date'],self.data_table['price2'], self.data_table['price1'])
                ax = plt.gca()
                ax = rf.loc[ox].plot(kind='line', x=rf.columns, ax=ax, grid=True,\
                                          color='blue', label=symbol+'s '+ox)
                ax.legend(loc=2)
                st.pyplot()

                
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
    
                
                # na=st.multiselect('Plot', rc)
                # st.line_chart(rf.loc[na])
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
            if opt== 'Discounted Cash Flow':
                dcf=fmp.discounted_cash_flow(apikey, symbol)
                dcfday=fmp.historical_daily_discounted_cash_flow(apikey, symbol,550)
                
                dcfdf=pd.DataFrame(dcfday)
                p=fmp.historical_price_full(apikey, symbol)
        if option == 'Technical analysis':
            st.subheader('Technical analysis')
            p=fmp.historical_price_full(apikey, symbol)
            prices=pd.DataFrame(p)
            # prices['date']=pd.to_datetime(prices['date'])
            prices=prices.set_index('date')
            # pricess=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo').json()['Time Series (5min)']
            
            # i=st.selectbox('Intraday'('4H', '1D', '1W', '1M'))
            
            sma50=fmp.technical_indicators(apikey, symbol,50, 'sma')
            sma100=fmp.technical_indicators(apikey, symbol,100, 'sma')
            sma200=fmp.technical_indicators(apikey, symbol,200, 'sma')
            
            fig = go.Figure(data=[go.Candlestick(x=prices.index, open=prices['open'],high= prices['high'], low=prices['low'], close=prices['close'])])
            
            st.plotly_chart(fig)
            
            
            
        if option == 'Stats':
            prices=pd.DataFrame(fmp.historical_price_full(apikey, symbol))
            rr=pd.DataFrame()
            prices=prices.set_index('date')
            rr['Values']=None
            rr=rr.transpose()
            rr['Mean']=np.mean(prices['changePercent'])
            rr['Median']=np.median(prices['changePercent'])
            rr['Standard deviation']=np.std(prices['changePercent'])
            rr['Beta']=fmp.company_profile(apikey, symbol)[0]['beta']
            rr['Skewness']=scipy.stats.skew(prices['changePercent'])
            rr['Kurtosis']=scipy.stats.kurtosis(prices['changePercent'])
            rr['Sharpe']=rr['Mean'] / rr['Standard deviation'] * np.sqrt(252)
            rr['VaR 95%']=np.percentile(prices['changePercent'],5)
            # ri=pd.DataFrame()
            # ri['Standard Deviation']=None
            # ri['Standard Deviation']=rr.loc(['Standard deviation'])
            # rr=rr.transpose()
            # counts, bins = np.histogram(prices['changePercent'], bins=range(0, 60, 5))
            # bins = 0.5 * (bins[:-1] + bins[1:])
            
            # fig = px.express.bar(x=bins, y=counts, labels={'x':f'Histogram for {symbol} daily returns', 'y':'count'})
            # fig.show()
            
            # fig = px.histogram(prices['changePercent'], x=f'Histogram for {symbol} daily returns')
            st.bar_chart(prices['changePercent'], 900, 400, True)
            
            fig=plt.figure()
            plt.hist(prices['changePercent'],bins=100)
            plt.title(symbol+' Daily Returns')
            # plt.xlabel()
            
            st.pyplot(fig)
            
            
            # rr['CVaR 95%'] = np.mean(prices['close'][prices['close'] <= rr['VaR 95%']])
            st.subheader('Risk Metrics')
            # st.markdown('Mean is:'+np.round(rr['Standard deviation'],4)+'')
            st.dataframe(rr)
        if option == 'Social Sentiment':
            lol=st.selectbox('Options', ('Stocktwits','Twitter', 'Reddit', 'Google Trends'))
            if lol == 'Stocktwits':
                nft='NFT'
                
        if option== 'News':
            select=st.selectbox('Options', ('News and articles','Press releases'))
            news=fmp.stock_news(apikey, symbol, 20)            
            press=fmp.press_releases(apikey, symbol, 20)
            if select == 'News and articles':
    
                st.image(news[0]['image'])
                for i in range(0,len(news)):
                    st.subheader('*'+news[i]['site']+'*'+' | '+news[i]['title'])
                    st.markdown(news[i]['text'])
                    st.markdown(news[i]['publishedDate']+' | '+news[i]['url'])
                    
            if select== 'Press releases':
                st.image(news[2]['image'])
                for i in range(0, len(press)):
                    st.subheader(press[i]['title']+' | '+press[i]['date'])
                    st.markdown(press[i]['text'])
                    
                    
        if option == 'SEC Filings':
            year=st.number_input('Year', 2013, 2022)
            Q=st.selectbox('Quarter', ('Q1', 'Q2', 'Q3', 'Q4'))
            filings=fmp.sec_filings(apikey, symbol)
            # for i in range(0,len(filings)):
            #     st.selectbox('Forms', (filings[i]['type']+' | '+filings[i]['acceptedDate']))
            # st.markdown("""
            # <embed src="https://www.sec.gov/Archives/edgar/data/88205/000008820522000013/0000088205-22-000013-index.htm" width="400" height="400">
            # """, unsafe_allow_html=True  )      
            # cf=fmp.form_13f(apikey, fmp.company_profile(apikey, symbol)['cik'].values())
            j=requests.get(f'https://fmpcloud.io/api/v4/financial-reports-json?symbol={symbol}'+f'&year={year}'+f'&period={Q}'+f'&apikey={apikey}').json()
            # t=requests.get(f'https://fmpcloud.io/api/v4/financial-reports-json?symbol={symbol}'+f'&year=2020&period=Q1&apikey={apikey}').text()
            opt=st.selectbox('Topic', j.keys())
            for i in range (0,len(j[opt])):
                opti=st.markdown(j[opt][i])
    
    
            g=fmp.financial_growth(apikey, symbol)
            fmp.commodities.commodities_list(apikey)
            # for element in j:
    
                # st.markdown(element)
                # for elements in j[element]:
                #     st.markdown(elements)
