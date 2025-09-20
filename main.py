# %%
# Connect with API

# %%
import json,os

with open('kaggle.json', 'r') as f:
    kaggle_creds = json.load(f)
    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("amanalisiddiqui/fraud-detection-dataset")

print("Path to dataset files:", path) 
print(os.listdir(path))


# %% [markdown]
# import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# %% [markdown]
# Read Dataset

# %%
file_path = os.path.join(path, 'AIML Dataset.csv')
df = pd.read_csv(file_path)
df.head()

# %% [markdown]
# Analyze Dataset

# %%
df.info()

# %%
df.columns

# %% [markdown]
# Count of Flagged Fraud in Dataset

# %%
df['isFraud'].value_counts()

# %%
df['isFlaggedFraud'].value_counts()

# %% [markdown]
# Check NA Values

# %%
df.isnull().sum()

# %%
df.shape

# %% [markdown]
# Percentage of Fraud in dataset

# %%
df['isFraud'].value_counts()[1] / df.shape[0] * 100

# %% [markdown]
# Visualizing data for more clarity

# %% [markdown]
# 1. Visualising "type" of transaction

# %%
df['type'].value_counts().plot(kind='bar', title='Transaction Type', color=['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral', 'lightgrey'   ])
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# 2. Fraud Rate by Type

# %%
fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fraud_by_type.plot(kind='bar', title='Fraud Rate by Transaction Type', color=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
plt.xlabel('Transaction Type')
plt.ylabel('Fraud Rate')
plt.show()

# %% [markdown]
# Amount Statistics

# %%
df['amount'].describe().astype(np.int64)

'''
%% [markdown]
Findings: 
count     6,362,620   → You have about 6.3 million transactions in total.
mean        179,861   → On average, each transaction is about $179k.
std         603,858   → The spread (variation) is HUGE — many transactions are far from the average.
min              0    → The smallest transaction is 0 (maybe test or failed transactions).
25%         13,389    → 25% of transactions are below $13k.
50%         74,871    → 50% are below $74k (this is the median).
75%        208,721    → 75% are below $208k.
max     92,445,516    → The largest transaction is ~92 million!

<!-- ------------------------------------------------------------------ -->
What this tells us :

a. Most transactions are relatively small (tens of thousands), but…
b. There are a few very large transactions (in the millions) that pull the average way up.
c. That’s why the mean (179k) is much bigger than the median (74k) → the data is skewed by huge values (outliers).

Why it matters in ML?:
# - ML models can get confused by extreme outliers (like that 92 million).
'''
# %% [markdown]
# Get Historgram of the above

# %%
sns.histplot(np.log1p(df['amount']), bins=50, kde=True)
plt.title('Log-Transformed Transaction Amount Distribution')
plt.xlabel('Log(Amount + 1)')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# If you try to make a normal histogram (without log), the chart gets squished:
#  - All the small transactions bunch up on the left.
#  - The giant $10M ones stretch the scale so much that the small ones almost disappear.
# 
# What np.log1p(df['amount']) does?
#  - It applies a logarithm to each amount (like zooming out on a big map).
#  - Logarithm makes big numbers shrink more than small ones.
# <!-- --------------------------------------------------------------------------------------->
# The purpose of log-transforming transaction amounts:
# 
# Without log:
# Imagine a classroom where 99 students are between 4–6 feet tall, but 1 student is 100 feet tall.
# If you make a height chart, that one giant makes the chart useless — everyone else looks the same squished at the bottom.
# 
# With log:
# You “shrink” the giant student so they’re more in line with the rest. Now you can see differences among the normal students clearly.
# 
# <!-- -------------------------------------------------------------------------------------------------->
# In ML terms:
#  - Real-world money data has a few giant values and lots of normal ones.
#  - The giant values can confuse the model.
#  - Log transformation makes the scale fairer, so the model learns patterns from all data, not just the outliers.

# %% [markdown]
# Let see transaction amount less than 50K is Fraud

# %%
sns.boxplot(data=df[df["amount"] < 50000], x='isFraud', y='amount')
plt.title('Transaction Amount by Fraud Status (Amounts < $50,000)')
plt.show()

# %% [markdown]
# This chart is like putting fraud and non-fraud transactions side by side to see how much money is usually involved

# %%
df.columns

# %%
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]


# %% [markdown]
# creating new columns from existing ones --> Feature Engineering

# %%
(df["balanceDiffOrig"] < 0).sum()

# %%
(df["balanceDiffDest"] < 0).sum()

# %% [markdown]
# What it means
# 
# -    oldbalanceOrg → the sender’s balance before the transaction.
# -    newbalanceOrig → the sender’s balance after the transaction.
#         -    Their difference = how much money left the sender’s account.
#         -    That’s saved in a new column balanceDiffOrig.
# -    oldbalanceDest → the receiver’s balance before the transaction.
# -    newbalanceDest → the receiver’s balance after the transaction.
#         -    Their difference = how much money arrived in the receiver’s account.
#         -    That’s saved in balanceDiffDest.

# %%
df.head(2)

# %% [markdown]
# we can note column "step" it is getting increased by 1, we can plot and drop it.

# %%
frauds_per_steps = df[df['isFraud'] == 1]["step"].value_counts().sort_index()
plt.plot(frauds_per_steps.index, frauds_per_steps.values, label='Fraudulent Transactions', color='red')
plt.xlabel('Step (time)')
plt.ylabel('Number of Fraudulent Transactions')
plt.title('Fraudulent Transactions Over Time')
plt.grid(True)
plt.show()

# %% [markdown]
# Step by step:
#     df[df['isFraud'] == 1]
# 
# Filter the dataset → keep only rows where the transaction is marked as fraud.
#     ["step"].value_counts()
# 
# Count how many fraudulent transactions happened in each time step.
#     (step is usually a unit of time in the dataset — e.g., 1 step = 1 hour).
#         .sort_index()
# 
# Sort the counts by time order (step 1, step 2, step 3 …).
#         plt.plot(...)
# 
# Plot fraud counts over time.
#     X-axis = time steps.
#     Y-axis = how many frauds happened at that time.
#     Line color = red for fraud.
# 
# Purpose of this plot:
#     To see when fraud happens in the timeline.
#     Helps answer questions like:
#     Are frauds evenly spread over time?
#     Do frauds spike at certain periods (e.g., night time, weekends, specific step ranges)?
#     Is fraud increasing over time (suggesting worsening attacks) or decreasing (maybe due to detection systems)?

# %%
df.drop(columns="step", inplace=True)

# %% [markdown]
# Let's go with Customer Wise, find out customers which makes the highest amount of transaction like top senders and top receivers 

# %%
df.columns

# %%
#Top senders
top_senders = df['nameOrig'].value_counts().head(10)

# %%
top_senders

# %%
#Top receivers
top_receivers = df['nameDest'].value_counts().head(10) 
top_receivers

# %% [markdown]
# Who are the top 10 people/accounts that got the most money transfers?
# It helps spot popular or suspicious receivers.

# %%
#Fraud making users
fraud_users = df[df['isFraud'] == 1]['nameOrig'].value_counts().head(10)
fraud_users

# %%
fraud_types = df[df['type'].isin(["TRANSFER", "CASH_OUT"])]
fraud_types.head()

# %%
fraud_types["type"].value_counts()

# %%
sns.countplot(data=fraud_types, x='type', hue='isFraud')
plt.title('Fraudulent Transactions in Transfer and Cash Out Types')
plt

# %%
corr = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Customers have zero amount after transfer

# %%
zero_after_transfer = df[
    (df["oldbalanceOrg"] > 0) &
    (df["newbalanceOrig"] == 0) &
    (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]

# %%
len(zero_after_transfer)

# %% [markdown]
# These are suspicious records

# %%
zero_after_transfer.head()

# %% [markdown]
# FEATURE SELECTION AND PREPRATION STEP

# %%
#import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# %%
df.head()

# %% [markdown]
# Drop some column

# %%
df_model = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# %%
df_model.head()

# %%
categorial_features = ['type']
numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# %%
y = df_model['isFraud']
X = df_model.drop('isFraud', axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop="first"), categorial_features)
    ],
    remainder= 'drop'
)

# %%
#Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced',max_iter=1000))])
    #------> Note if we will not set class_weight='balanced' the model 
    # will be biased towards non-fraud class, since the dataset is highly imbalanced. 99% of the transactions are non-fraudulent.


# %%
pipeline.fit(X_train, y_train)

# %%
y_pred = pipeline.predict(X_test)

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# Findings: 
# What the numbers mean
#     Class 0 → Not Fraud (the majority)
# 
#         Precision: 1.00 → Almost every time the model predicts "Not Fraud," it’s correct.
# 
#         Recall: 0.94 → It catches 94% of the actual non-fraud cases (misses 6%).
# 
#         F1: 0.97 → Strong overall performance here.
# 
#     Class 1 → Fraud (the minority)
# 
#         Precision: 0.02  → Only 2% of the transactions flagged as fraud are actually fraud → huge false positive problem.
# 
#         Recall: 0.95 → Model catches 95% of all frauds.
# 
#         F1: 0.04 → Very low overall balance between precision & recall.
# 
#     Overall
# 
#         Accuracy: 0.94 → Seems high, but misleading because the dataset is imbalanced (most transactions are non-fraud).
# 
#         Macro avg (balanced view): Precision = 0.51, Recall = 0.95, F1 = 0.51.
# 
#         Weighted avg (skewed by majority class): Looks great (0.97 F1), but that’s because non-fraud dominates.
# 
#         What this really says
# 
#             The model is super sensitive → it calls nearly every possible fraud a fraud.
# 
#             That’s why recall for fraud is high (0.95) but precision is terrible (0.02).
# 
#         In practice:
# 
#             It catches almost all frauds 
# 
#             But it falsely flags a huge number of normal transactions

# %%
array = confusion_matrix(y_test, y_pred)
array

# %%
#Accuracy
pipeline.score(X_test, y_test) * 100

# %%
import joblib
joblib.dump(pipeline, 'fraud_detection_model.pkl')


