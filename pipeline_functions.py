# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('cc_institution_details.csv')
df2 = pd.read_csv('campus_recruitment.csv')

# %%
def coerce_numeric(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
# %%
def drop_missing(df):
    return df.dropna().copy()
# %%
def select_college_vars(df):
    cols = ["aid_value", "med_sat_value", "fte_value", "ft_pct", "grad_100_value"]
    return df[cols].copy()
# %%
def create_grad_target(df):
    median_grad = df["grad_100_value"].median()
    df = df.copy()
    df["high_4yr_grad"] = (df["grad_100_value"] >= median_grad).astype(int)
    df = df.drop(columns=["grad_100_value"])
    return df
# %%
def college_completion_pipeline(df):
    df = select_college_vars(df)
    df = coerce_numeric(df, df.columns)
    df = drop_missing(df)

    df = create_grad_target(df)

    X = df.drop(columns=["high_4yr_grad"])
    y = df["high_4yr_grad"]

    X_train, X_tune, X_test, y_train, y_tune, y_test = split_data(X, y)

    numeric_cols = X.columns.tolist()
    X_train, X_tune, X_test = scale_numeric(
        X_train, X_tune, X_test, numeric_cols
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test

# %%
def select_recruitment_vars(df):
    cols = ["hsc_p", "hsc_s", "ssc_p", "degree_p", "degree_t", "workex", "status"]
    return df[cols].copy()
# %%
def create_placement_target(df):
    df = df.copy()
    df["placed"] = (df["status"] == "Placed").astype(int)
    return df.drop(columns=["status"])
# %%
def encode_categoricals(df, cat_cols):
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %%
def campus_recruitment_pipeline(df):
    df = select_recruitment_vars(df)
    df = coerce_numeric(df, ["hsc_p", "ssc_p", "degree_p"])
    df = drop_missing(df)

    df = create_placement_target(df)

    df = encode_categoricals(
        df,
        cat_cols=["hsc_s", "degree_t", "workex"]
    )

    X = df.drop(columns=["placed"])
    y = df["placed"]

    X_train, X_tune, X_test, y_train, y_tune, y_test = split_data(X, y)

    numeric_cols = ["hsc_p", "ssc_p", "degree_p"]
    X_train, X_tune, X_test = scale_numeric(
        X_train, X_tune, X_test, numeric_cols
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test
