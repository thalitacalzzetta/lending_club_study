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


df = pd.read_csv(
        "loan_data.csv",
        sep=None
        )
df_model = df[["revol.util", "fico", "int.rate"]]
df_model.columns = ['revol_util', 'fico', 'int_rate']

df_model['int_rate'] = df_model['int_rate'] * 100

df_sample = df_model.sample(n=1000, random_state = 50)
print(df_sample)
print(df_sample.isna().sum())

# Scatter plot - interest rate vs revol_util

plt.figure()
plt.scatter(df_sample["revol_util"], df_sample["int_rate"])
plt.xlabel("revol_util")
plt.ylabel("interest_rate")
plt.show()

# Scatter plot - interest rate vs Fico

plt.figure()
plt.scatter(df_sample["fico"], df_sample["int_rate"])
plt.xlabel("fico")
plt.ylabel("interest_rate")
plt.show()


x = df_sample[["revol_util", "fico"]]
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








