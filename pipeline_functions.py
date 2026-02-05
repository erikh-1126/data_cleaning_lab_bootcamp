# %%
# Import packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Convert the columns to numeric data
def coerce_numeric(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
    
# %%
# Drop missing values
def drop_missing(df):
    return df.dropna().copy()

# %%
# Split the data into training, tuning, and testing data
def split_data(X, y, test_size=0.4, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, stratify=y,random_state=random_state)
    X_tune, X_test, y_tune, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)
    return X_train, X_tune, X_test, y_train, y_tune, y_test

# %%
# Scale numeric variables
def scale_numeric(X_train, X_tune, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_tune[numeric_cols] = scaler.transform(X_tune[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_tune, X_test

# %%
# College Completion:
df = pd.read_csv('cc_institution_details.csv')

# %%
# Select variables
def select_college_vars(df):
    cols = ["aid_value", "med_sat_value", "fte_value", "ft_pct", "grad_100_value"]
    return df[cols].copy()
    
# %%
# Make target variable
def create_grad_target(df):
    median_grad = df["grad_100_value"].median()
    df = df.copy()
    df["high_4yr_grad"] = (df["grad_100_value"] >= median_grad).astype(int)
    df = df.drop(columns=["grad_100_value"])
    return df
    
# %%
# Make the pipeline
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
# Campus Recruitment:

# %%
# Select variables
def select_recruitment_vars(df):
    cols = ["hsc_p", "hsc_s", "ssc_p", "degree_p", "degree_t", "workex", "status"]
    return df[cols].copy()
    
# %%
# Create target variable
def create_placement_target(df):
    df = df.copy()
    df["placed"] = (df["status"] == "Placed").astype(int)
    return df.drop(columns=["status"])
    
# %%
# Encode categorical variables
def encode_categoricals(df, cat_cols):
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %%
# Make pipeline
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
