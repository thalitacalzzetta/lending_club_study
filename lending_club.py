#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:17:30 2026

@author: tcalzzetta
"""

# %% 1. Important Libraries

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% 2. Load Data

df = pd.read_csv(
        "loan_data.csv",
        sep=None
        )

df_model = df[[
    "int.rate",
    "revol.util", 
    "fico",
    "installment",
    "log.annual.inc",
    "dti",
    "days.with.cr.line",
    "revol.bal",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
    ]]

df_model.columns = [
    'int_rate',
    'revol_util',
    'fico',
    'installment',
    'log_annual_inc',
    'dti',
    'days_credit_line',
    'revolv_bal',
    'inq_last_6_months',
    'delinq_2yrs',
    'public_rec',
    ]

df_model['int_rate'] = df_model['int_rate'] * 100

# Annual Income is log, so we will annalyze both log and annual income to understand why should we use log.

df_model["annual_income"] = np.exp(df_model["log_annual_inc"]) / 1000

df_sample = df_model.sample(n=1000, random_state = 50)
print(df_sample)
print(df_sample.isna().sum())

# %% 3. Exploring Data Analysis - Relationships with Interest Rate

# 3.1 - Interest Rate vs Revolving Utilization Rate

plt.figure()
plt.scatter(df_sample["revol_util"], df_sample["int_rate"])
plt.xlabel("Revolving Line Utilization (%)")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Revolving Credit Utilization Rate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.2 - Interest Rate vs Fico Score

plt.figure()
plt.scatter(df_sample["fico"],df_sample["int_rate"])
plt.xlabel("Fico Score")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Fico Score") 
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.3 - Interest Rate vs Installment

plt.figure()
plt.scatter(df_sample["installment"], df_sample["int_rate"])
plt.xlabel("Installment - Monthly")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Monthly Installments")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.4 - Interest Rate vs Annual Income 

plt.figure()
plt.scatter(df_sample["annual_income"], df_sample["int_rate"])
plt.xlabel("Annual Income ($ 1000)")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Annual Income")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.5 - Interest Rate vs Annual Income(log)

plt.figure()
plt.scatter(df_sample["log_annual_inc"], df_sample["int_rate"])
plt.xlabel("Annual Income (log)")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Annual Income")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.6 - Interest Rate vs Debt-to-Income Ratio

plt.figure()
plt.scatter(df_sample["dti"], df_sample["int_rate"])
plt.xlabel("Debt-to-Income")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Debt-to-Income Ratio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.7 - Interest Rate vs Days with Credit Line

plt.figure()
plt.scatter(df_sample["days_credit_line"], df_sample["int_rate"])
plt.xlabel("Days with Credit Line")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Days with Credit Line")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# 3.8 - Interest Rate vs Revolving Balance

plt.figure()
plt.scatter(df_sample["revolv_bal"], df_sample["int_rate"])
plt.xlabel("Revolving Balance ($)")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Revolving Balance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# 3.9 - Interest Rate vs Inquiries in the Last 6 Months

plt.figure()
sns.boxplot(data=df_sample, x="inq_last_6_months", y="int_rate")
plt.xlabel("Credit Inquiries - last 6 Months")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Credit Inquiries")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 3.10 - Interest Rate vs Delinquencies in the Last 2 Years

plt.figure()
sns.boxplot(data=df_sample, x="delinq_2yrs", y="int_rate")
plt.xlabel("Delinquencies in the Last 2 Years")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Delinquencies")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# 3.11 - Interest Rate vs Public Records

plt.figure()
sns.boxplot(data=df_sample, x="public_rec", y="int_rate")
plt.xlabel("Number of Derogatory Public Records")
plt.ylabel("Interest Rate (%)")
plt.title("Interest Rate vs Public Records")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# %% 4. Correlation Matrix (Pearson Correlations)

correlation_matrix = (df_sample.corr()).round(2)
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix - Lending Club")
plt.show()

# %% 5. Modeling

x = df_sample[[
    'public_rec',
    'delinq_2yrs',
    'days_credit_line',
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]
x = sm.add_constant(x)

y = df_sample["int_rate"]

model = sm.OLS(y,x).fit()
print(model.summary())

# Predicting and calculating residuals

df_sample["predicted_ir"] = model.predict(x)
df_sample["residuals"] = df_sample["int_rate"] - df_sample["predicted_ir"]
print(df_sample)

plt.figure()
plt.scatter(df_sample["predicted_ir"], df_sample["residuals"])
plt.axhline(y=0)
plt.xlabel("Predicted Interest Rate")
plt.ylabel("Residuals")
plt.show()

plt.hist(df_sample["residuals"], bins=100)

# %% 6. Testing the Model excluding one variable per time (variables with the highest p-values)

p_values = round(model.pvalues,2)
p_values.sort_values(ascending=False)

# %% 6.1 - Removing public_rec

x_pub = df_sample[[
    'delinq_2yrs',
    'days_credit_line',
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_pub = sm.add_constant(x_pub)

model_pub = sm.OLS(y, x_pub).fit()
print(model_pub.summary())

round(model_pub.pvalues.sort_values(ascending=False),2)

# %% 6.2 - Removing delinq_2yrs

x_del = df_sample[[
    'days_credit_line',
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_del = sm.add_constant(x_del)

model_del = sm.OLS(y, x_del).fit()
print(model_del.summary())

round(model_del.pvalues.sort_values(ascending=False),2)

# %% 6.3 - Removing days_credit_line

x_cred = df_sample[[
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_cred = sm.add_constant(x_cred)

model_cred = sm.OLS(y, x_cred).fit()
print(model_cred.summary())

round(model_cred.pvalues.sort_values(ascending=False),2)


# %% 6.4 - Removing log_annual_inc

x_log = df_sample[[
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_log = sm.add_constant(x_log)

model_log = sm.OLS(y, x_log).fit()
print(model_log.summary())

round(model_log.pvalues.sort_values(ascending=False),2)


# %% 6.5 - Removing revol_util

x_util = df_sample[[
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_util = sm.add_constant(x_util)

model_util = sm.OLS(y, x_util).fit()
print(model_util.summary())

round(model_util.pvalues.sort_values(ascending=False),2)

# %% 6.6 - Removing inq_last_6_months

x_inq = df_sample[[
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_inq = sm.add_constant(x_inq)

model_inq = sm.OLS(y, x_inq).fit()
print(model_inq.summary())

round(model_inq.pvalues.sort_values(ascending=False),2)

# %% Creating list with p-values after each remodeling

dic_models = {
    "Original Model": model,
    "Model 1": model_pub,
    "Model 2":model_del,
    "Model 3":model_cred,
    "Model 4":model_log,
    "Model 5": model_util,
    "Model 6": model_inq
    }

p_values_dic = {}

for i, model_p in dic_models.items():
  list_pvalues = model_p.pvalues
  p_values_dic[i] = list_pvalues
    
df_pvalues = pd.DataFrame(
    p_values_dic)

df_pvalues = round(df_pvalues.reindex(x.columns),3)
print(df_pvalues)

# %% Creating the comparative table


# 1. Creating the list with all statistic metrics (using the dic_models)

models = []

for key, regression in dic_models.items():
    r_squared = (regression.rsquared)
    r_squared_adj = (regression.rsquared_adj)
    f_value = (regression.fvalue)
    f_pvalue = (regression.f_pvalue)
    SSE_model = (regression.ssr)
    
    statistics = (
        key,
        r_squared,
        r_squared_adj,
        f_value,
        f_pvalue,
        SSE_model
        )
    
    models.append(
        statistics
        )

columns_models = [
    "Model",
    "R-squared",
    "R-squared Adjusted",
    "F-statistic",
    "P( F-statistic)",
    "SSE"
    ]

df_models = pd.DataFrame(
    models,
    columns = columns_models,
    )
df_models = df_models.set_index("Model")
print(df_models)
