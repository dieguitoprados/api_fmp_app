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



apikey='d60d2f087ecf05f94a3b9b3df34310a9'
symbol= st.text_input("stock", '')
if symbol != '':
    
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
    st.subheader("Balance Sheet")
    st.dataframe(balance_sheet)
    
    st.subheader("Income Statement")
    st.dataframe(income_statement)
    
    st.subheader("Cashflow Statement")    
    st.dataframe(cashflow_statement)
