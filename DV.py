import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# --- 1. Re-simulate the Dataset (as in previous turn for continuity) ---
np.random.seed(42) # for reproducibility

num_customers = 1000

data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 70, num_customers),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], num_customers, p=[0.48, 0.50, 0.02]),
    'MonthlyCharges': np.random.uniform(20, 150, num_customers).round(2),
    'TotalCharges': np.random.uniform(50, 5000, num_customers).round(2),
    'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_customers, p=[0.6, 0.2, 0.2]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], num_customers, p=[0.3, 0.25, 0.25, 0.2]),
    'TenureMonths': np.random.randint(1, 72, num_customers), # How long they've been a customer
    'Churn': np.random.choice([0, 1], num_customers, p=[0.8, 0.2]), # 0 = No Churn, 1 = Churn
    'SignUpDate': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1000, num_customers), unit='D')
}

df = pd.DataFrame(data)

# Introduce some missing values and anomalies for demonstration (and then clean them)
df.loc[np.random.choice(df.index, 30, replace=False), 'TotalCharges'] = np.nan # 30 missing values
df.loc[np.random.choice(df.index, 5, replace=False), 'Age'] = 150 # Outliers in Age
df.loc[np.random.choice(df.index, 2, replace=False), 'MonthlyCharges'] = -50 # Negative values (anomaly)
df.loc[np.random.choice(df.index, 10, replace=False), 'ContractType'] = 'month-to-month' # Inconsistent casing

# --- 2. Data Cleaning and Preprocessing (as identified in previous EDA) ---

# Handle 'TotalCharges' missing values (e.g., impute with median)
# For simplicity, let's fill NaN with the median. In a real project, consider context.
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Handle 'Age' outliers (e.g., cap at a reasonable max or remove)
# Let's cap age at 90, assuming no customers are older than that.
df['Age'] = df['Age'].apply(lambda x: 90 if x > 90 else x)

# Handle 'MonthlyCharges' negative values (e.g., replace with median or 0)
# Let's replace negative values with 0, assuming they might be errors or free trials.
df['MonthlyCharges'] = df['MonthlyCharges'].apply(lambda x: max(0, x))

# Standardize 'ContractType' casing
df['ContractType'] = df['ContractType'].str.capitalize().replace({'Month-to-month': 'Month-to-month'})

print("--- Dataset after basic cleaning ---")
print(df.head())
print(df.isnull().sum()) # Verify no more NaNs in TotalCharges
print(df['Age'].max()) # Verify age cap
print(df['MonthlyCharges'].min()) # Verify no negative charges
print(df['ContractType'].value_counts()) # Verify standardized casing
print("\n" + "="*50 + "\n")


# --- 3. Data Visualization and Storytelling ---

print("--- Visualizing Key Insights ---")

# Visualization 1: Overall Churn Rate
plt.figure(figsize=(6, 6))
churn_counts = df['Churn'].value_counts()
churn_labels = ['No Churn (80%)', 'Churn (20%)'] # Using the expected distribution
churn_colors = ['#66b3ff', '#ff9999'] # Blue for no churn, red for churn
plt.pie(churn_counts, labels=churn_labels, autopct='%1.1f%%', startangle=90, colors=churn_colors,
        wedgeprops={'edgecolor': 'black'})
plt.title('Overall Customer Churn Rate', fontsize=16)
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
print("Insight: The company has a churn rate of approximately 20%. This is a critical metric to monitor and reduce.")


# Visualization 2: Churn by Contract Type (Stacked Bar Chart)
# Calculate churn rate by contract type
churn_by_contract = df.groupby('ContractType')['Churn'].value_counts(normalize=True).unstack().fillna(0)
churn_by_contract.columns = ['No Churn', 'Churn'] # Rename columns for clarity

plt.figure(figsize=(9, 6))
churn_by_contract.plot(kind='bar', stacked=True, color=['#88CCEE', '#CC6677'], edgecolor='black')
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Proportion of Customers', fontsize=12)
plt.xticks(rotation=0)
plt.lege
