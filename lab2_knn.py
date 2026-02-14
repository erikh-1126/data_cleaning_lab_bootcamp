"""
Graduation Lab: Week 6


Instructions:

Let's build a kNN model using the college completion data. 
The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

You've done most of the hard work already, so you should be ready to move forward with building your model. 

1. Use the question/target variable you submitted and 
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values, 
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion matrix look the same at the same threshold 
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
concerns or positive elements do you have about the model as it relates to your question? 

6. Create two functions: One that cleans the data & splits into training|test and one that 
allows you to train and test the model with different k and threshold values, then use them to 
optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.) 

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7. 

"""

## Step 1: Question/Target Variable:
# (Changed so that the problem is a classification problem): Do public, private non-profit, or private for-profit institutions give out more financial aid to students?

# %%
import pandas as pd
import numpy as np
df = pd.read_csv('cc_institution_details.csv')
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# %%
## Step 2: Building the kNN model with 3 nearest neighbors
features = ["aid_value", "aid_percentile"]
def prepare_college_data(df, features, target_col="control"):

    df = df[features + [target_col]].copy()

    # Convert features to numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing values
    df = df.dropna()

    X = df[features]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # Scale predictors
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# %%
## Step 3: Building the dataframe with test values, target values, predicted values, and probabilities
def run_knn(X_train, X_test, y_train, y_test, k=3):
    # Making the model and fitting it to the training data
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    # Making predictions on the test data
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    # Making the results dataframe
    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "probability_max_class": probs.max(axis=1)
    })
    # Building a confusion matrix for later
    cm = confusion_matrix(y_test, preds)

    return model, results, cm
# %%
# Running the functions to prepare the data and run the model
X_train, X_test, y_train, y_test = prepare_college_data(
    df,
    features,
    target_col="control"
)

model, results_df, cm = run_knn(
    X_train, X_test, y_train, y_test, k=3
)

print(results_df.tail())
# %%
## Step 4: Would adjusting the k hyperparameter change the threshold function and confusion matrix at the same threshold levels?
# When adjusting the k hyperparameter, the threshold function would more than likely change a little, as the boundaries of the function would have shifted.
# The confusion matrix would also likely change at the same threshold levels, as the predicted values would likely change with a different k value.
# Whether it is more or less accurate would depend on the data and the specific k values chosen.

# %%
## Step 5: Evaluation with the confusion matrix

print(cm)

# When looking at my confusion matrix, it did a very good job at identifying which schools belong in which category.
# I believe that using both categories of the amount of aid and the percentile of the aid proved to be a big identifier for the model.
# Even though the model did quite a good job, it wasn't perfect, as its errors per column or row ranged from 5 to 22.
# In terms of the question, I made sure to outline the classifications, which was clearly relayed by the efficiency of the model.
# %%
## Step 6: Optimizing for different k levels

for k in [3,5,7,9]:
    _, _, cm = run_knn(X_train, X_test, y_train, y_test, k=k)
    print(f"\nConfusion matrix for k={k}")
    print(cm)

# %%
## Step 7: Evaluating the variation of k levels:

# When looking at the variance of the confusion matrix at different k levels, while there is change, it doesn't exactly improve the efficiency of the model.
# In fact, the total accuracy decreases over all three categories.

# %%
## Step 8: A New Target Variable (level)
# Do two-year or four-year universities give out the most aid?
X_train, X_test, y_train, y_test = prepare_college_data(
    df,
    features,
    target_col="level"
)

model, results_df, cm = run_knn(
    X_train, X_test, y_train, y_test, k=3
)

print(results_df.tail())
print(cm)
# %%
