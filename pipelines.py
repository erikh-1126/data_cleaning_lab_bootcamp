# Step 1: Review Datasets and Come Up with Question

# College Completion: Do schools that give out more aid have a higher percentage of students graduating in 4 years?
# Campus Recruitment: Do different specializations require higher percentages for higher secondary education?

# Step 2: Business Metric and Data Prep
# College Completion:
# Generic Question: Do schools that give out more aid have a higher percentage of students graduating in 4 years?
# Business Metric: aid_percentile (the amount of aid given out by the school compared to other schools)

# Data Prep:
#%%
import os
os.path.getsize("cc_institution_details.csv")


#%%
# Import packages and dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cc_institution_details.csv')
cols = ["aid_value", "med_sat_value", "fte_value", "ft_pct", "grad_100_value"]

# %%
# Include only relevant variables
df = df[cols].copy()
# Correct variable types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")



# %%
# Drop missing values
df = df.dropna()


# %%
# Make target variable
median_grad = df["grad_100_value"].median()
df["high_4yr_grad"] = (df["grad_100_value"] >= median_grad).astype(int)
df = df.drop(columns=["grad_100_value"])

# %%
# Calculate prevalence of target variable
prevalence = df["high_4yr_grad"].mean()
print(f"Prevalence of high four-year graduation: {prevalence:.3f}")

# %%
# Separate features from the target
X = df.drop(columns=["high_4yr_grad"])
y = df["high_4yr_grad"]

# %%
# Normalize variables
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)


# %%
# Split the data into training, tuning, and testing data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

X_tune, X_test, y_tune, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)
# %%
# Sanity check
print("Train size:", X_train.shape)
print("Tune size:", X_tune.shape)
print("Test size:", X_test.shape)

print("Train prevalence:", y_train.mean())
print("Tune prevalence:", y_tune.mean())
print("Test prevalence:", y_test.mean())

# Campus Recruitment:
# Generic Question: Do different specializations require higher percentages for higher secondary education?
# Business Metric: hsc_p (Higher Secondary Education percentage)

# Data Prep:
#%%
os.path.getsize("placement_data_full_class.csv")

# %%
# Import dataset
df2 = pd.read_csv("placement_data_full_class.csv")
cols2 = ["hsc_p", "hsc_s", "ssc_p", "degree_p", "degree_t", "workex", "status"]

# %%
# Include only relevant variables
df2 = df2[cols2].copy()
num_cols = ["hsc_p", "ssc_p", "degree_p"]
cat_cols = ["hsc_s", "degree_t", "workex", "status"]

for col in num_cols:
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

for col in cat_cols:
    df2[col] = df2[col].astype("category")
    
# %%
# Drop missing values
df2 = df2.dropna()

# %%
# Make target variable
df2["placed"] = (df2["status"] == "Placed").astype(int)
df2 = df2.drop(columns=["status"])

# %%
# Calculate prevalence of target variable
prevalence = df2["placed"].mean()
print(f"Placement prevalence: {prevalence:.3f}")

# %%
# Encode categorical variables
df2_encoded = pd.get_dummies(
    df2,
    columns=["hsc_s", "degree_t", "workex"],
    drop_first=True
)

# %%
# Separate features from the target
X2 = df2_encoded.drop(columns=["placed"])
y2 = df2_encoded["placed"]

# %%
# Normalize variables
scaler = StandardScaler()

X2[num_cols] = scaler.fit_transform(X2[num_cols])

# %%
# Split the data into training, tuning, and testing data
X2_train, X2_temp, y2_train, y2_temp = train_test_split(
    X2,
    y2,
    test_size=0.4,
    stratify=y2,
    random_state=42
)

X2_tune, X2_test, y2_tune, y2_test = train_test_split(
    X2_temp,
    y2_temp,
    test_size=0.5,
    stratify=y2_temp,
    random_state=42
)

# %%
# Sanity check
print("Train size:", X2_train.shape)
print("Tune size:", X2_tune.shape)
print("Test size:", X2_test.shape)

print("Train prevalence:", y2_train.mean())
print("Tune prevalence:", y2_tune.mean())
print("Test prevalence:", y2_test.mean())

# Step 3: Instincts and Concerns

# College Completion: Based on my instincts, I believe that schools that provide more financial aid will have higher graduation rates within 4 years. 
# Financial aid can alleviate the financial burden on students, allowing them to focus more on their studies and complete their degrees on time. 
# Along with that, I do believe that the data provided can confirm that there is a positive correlation between the amount of aid given and the graduation rates.
# However, I am concerned about potential confounding factors such as the quality of education, student support services, and socioeconomic backgrounds of the students that might also influence graduation rates.


# Campus Recruitment: My instinct that different job specializations do require higher percentages for higher secondary education, and that Science may requite higher percentages compared to Commerce and Arts.
# The data provided can help confirm this by analyzing the placement rates across different specializations and their corresponding higher secondary education percentages.
# However, I am concerned the self-selection bias in this dataset, as students who choose certain specializations may inherently have different academic capabilities or motivations that could influence their placement outcomes.
