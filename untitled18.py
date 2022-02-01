# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 21:20:21 2022

@author: diegu
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import fmpsdk as fmp



apikey='d60d2f087ecf05f94a3b9b3df34310a9'
symbol= st.text_input("stock", '')
data_type= 'datatype=csv'
header='https://fmpcloud.io/api/v3/'
# sector=


balance_sheets=fmp.company_valuation.balance_sheet_statement(apikey, symbol)
income_statements=fmp.company_valuation.income_statement(apikey, symbol)
cashflow_statements=fmp.company_valuation.cash_flow_statement(apikey, symbol)

# tablas balances pyg y cf

balance_sheet=pd.DataFrame()
balance_sheet['index']=balance_sheets[0].keys()
balance_sheet=balance_sheet.set_index('index')

for i in range(0,len(balance_sheets)):

    balance_sheet[balance_sheets[i]['date']]=balance_sheets[i].values()
balance_sheet=balance_sheet[balance_sheet.columns[::-1]]
balance_sheet=balance_sheet.astype(str)

income_statement=pd.DataFrame()
income_statement['index']=income_statements[0].keys()
income_statement=income_statement.set_index('index')
for i in range(0,len(income_statements)):
    
    income_statement[income_statements[i]['date']]=income_statements[i].values()
income_statement=income_statement[income_statement.columns[::-1]]
income_statement=income_statement.astype(str)

cashflow_statement=pd.DataFrame()
cashflow_statement['index']=cashflow_statements[0].keys()
cashflow_statement=cashflow_statement.set_index('index')
for i in range(0,len(cashflow_statements)):

    cashflow_statement[cashflow_statements[i]['date']]=cashflow_statements[i].values()
cashflow_statement=cashflow_statement[cashflow_statement.columns[::-1]]
cashflow_statement=cashflow_statement.astype(str)

# analisis horizontal
hori_balance_sheet=fmp.company_valuation.balance_sheet_statement_growth(apikey, symbol)
hori_income_statement=fmp.company_valuation.income_statement_growth(apikey, symbol)
hori_cashflow_statement=fmp.company_valuation.cash_flow_statement_growth(apikey, symbol)
balance_sheet_hori=pd.DataFrame()
income_statement_hori=pd.DataFrame()
cashflow_statement_hori=pd.DataFrame()
balance_sheet_hori['index']=hori_balance_sheet[0].keys()
balance_sheet_hori=balance_sheet_hori.set_index('index')
income_statement_hori['index']=hori_income_statement[0].keys()
income_statement_hori=income_statement_hori.set_index('index')
cashflow_statement_hori['index']=hori_cashflow_statement[0].keys()
cashflow_statement_hori=cashflow_statement_hori.set_index('index')

for i in range(0,len(hori_balance_sheet)):

    balance_sheet_hori[hori_balance_sheet[i]['date']]=hori_balance_sheet[i].values()
balance_sheet_hori=balance_sheet_hori[balance_sheet_hori.columns[::-1]]
balance_sheet_hori=balance_sheet_hori.astype(str)

for i in range(0,len(hori_income_statement)):

    income_statement_hori[hori_income_statement[i]['date']]=hori_income_statement[i].values()
income_statement_hori=income_statement_hori[income_statement_hori.columns[::-1]]
income_statement_hori=income_statement_hori.astype(str)

for i in range(0,len(hori_cashflow_statement)):

    cashflow_statement_hori[hori_cashflow_statement[i]['date']]=hori_cashflow_statement[i].values()
cashflow_statement_hori=cashflow_statement_hori[cashflow_statement_hori.columns[::-1]]
cashflow_statement_hori=cashflow_statement_hori.astype(str)


ratios=fmp.company_valuation.financial_ratios_ttm(apikey, symbol)

news=fmp.company_valuation.stock_news(apikey, symbol)



st.image('https://finviz.com/chartashx?t='+symbol)
