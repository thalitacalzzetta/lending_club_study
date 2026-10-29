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

model_1 = sm.OLS(y, x_pub).fit()
print(model_1.summary())

round(model_1.pvalues.sort_values(ascending=False),2)

# %% 6.2 - Removing delinq_2yrs

x_2 = df_sample[[
    'days_credit_line',
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_2 = sm.add_constant(x_2)

model_2 = sm.OLS(y, x_2).fit()
print(model_2.summary())

round(model_2.pvalues.sort_values(ascending=False),2)

# %% 6.3 - Removing days_credit_line

x_3 = df_sample[[
    'log_annual_inc',
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_3 = sm.add_constant(x_3)

model_3 = sm.OLS(y, x_3).fit()
print(model_3 .summary())

round(model_3 .pvalues.sort_values(ascending=False),2)


# %% 6.4 - Removing log_annual_inc

x_4 = df_sample[[
    'revol_util',
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_4 = sm.add_constant(x_4)

model_4 = sm.OLS(y, x_4).fit()
print(model_4.summary())

round(model_4.pvalues.sort_values(ascending=False),2)


# %% 6.5 - Removing revol_util

x_5 = df_sample[[
    'inq_last_6_months',
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_5 = sm.add_constant(x_5)

model_5 = sm.OLS(y, x_5).fit()
print(model_5.summary())

round(model_5.pvalues.sort_values(ascending=False),2)

# %% 6.6 - Removing inq_last_6_months

x_6 = df_sample[[
    'revolv_bal',
    'dti',
    'installment',
    'fico'
    ]]

x_6 = sm.add_constant(x_6)

model_6 = sm.OLS(y, x_6).fit()
print(model_6.summary())

round(model_6.pvalues.sort_values(ascending=False),2)

# %% Creating list with p-values after each remodeling

dic_models = {
    "Original Model": model,
    "Model 1": model_1,
    "Model 2":model_2,
    "Model 3":model_3,
    "Model 4":model_4,
    "Model 5": model_5,
    "Model 6": model_6
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
    "F-statistic",
    "P( F-statistic)",
    "SSE"
    ]

df_models = pd.DataFrame(
    models,
    columns = columns_models,
    )
df_models = round(df_models.set_index("Model"),5)
print(df_models)


    

# Considering the possible correlation between int_rate and installment, I will create a new model before applying log.

# %% Creating new model to evaluate a possible correlation between int_rate and installment.

# 1.1 - Model 6 excluding installment.

x_7 = df_sample[[
    'revolv_bal',
    'dti',
    'fico'
    ]]

x_7 = sm.add_constant(x_7)

model_7 = sm.OLS(y, x_7).fit()
print(model_7.summary())

round(model_7.pvalues.sort_values(ascending=False),2)

# %% Choosing 4 models to compare.

# 1. Analyzing residuals both "Original Model", "model 4" and "model 6".

# 1.1 - Predicted and Residuals

df_residuals = pd.DataFrame(df_sample["int_rate"])

df_models_residuals = {
    "Original Model": [model, x],
    "Model 4": [model_4, x_4],
    "Model 6": [model_6, x_6],
    "Model 7": [model_7, x_7]}

for name, (model_version, x_version) in df_models_residuals.items():
    predicted_models = model_version.predict(x_version)
    
    df_residuals[f"{name} - predicted"] = predicted_models
    
    df_residuals[f"{name} - residuals"] = df_residuals["int_rate"] - predicted_models
    
print(df_residuals)

# 1.2 - Scatter Plot and histogram - predicted vs residuals

models_names = ["Original Model", "Model 4", "Model 6", "Model 7"]

for name in models_names:
    
    predicted = df_residuals[f"{name} - predicted"]
    residuals = df_residuals[f"{name} - residuals"]
    
    plt.figure()
    plt.scatter(predicted, residuals)
    plt.axhline(y=0)
    plt.xlabel(f"predicted - {name}")
    plt.ylabel(f"residuals - {name}")
    plt.show()
    
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.title(f"Residual Distribution - {name}")
    plt.show()

# Considering that installment is not in the model 7, the model performs better (because revolv_util is now relevant).
# Before validate the models and choose the one that best predicts the interest_rate, I want to build models with categorical variables,
# that based on my experience, can have influence in the interest rate.

# %% Working with categorical variables.

# 1. Credit purpose.

# This is the categorical variable available in the dataset. We will build a model with categorical variables and understand the model.

df_sample['purpose'] = df.loc[df_sample.index, 'purpose']

# 1.1 - Creating the categories.

categories = df_sample["purpose"].value_counts()

df_sample["purpose_cat"] = pd.Categorical(df_sample["purpose"], categories=categories)
# i need to create dummy variables. 