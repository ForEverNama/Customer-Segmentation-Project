import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_customers = 500

# Generate synthetic demographic and behavioral data
data = {
    'CustomerID': range(1001, 1001 + n_customers),
    'Age': np.random.randint(18, 70, size=n_customers),
    'Annual_Income_k': np.random.randint(20, 140, size=n_customers),
    'Spending_Score': np.random.randint(1, 100, size=n_customers),
    'Total_Purchases': np.random.randint(5, 80, size=n_customers)
}

df = pd.DataFrame(data)

# Inject artificial patterns to help clustering succeed cleanly
df.loc[df['Annual_Income_k'] > 90, 'Spending_Score'] = np.random.randint(70, 100, size=len(df[df['Annual_Income_k'] > 90]))
df.loc[df['Age'] > 55, 'Spending_Score'] = np.random.randint(10, 40, size=len(df[df['Age'] > 55]))

df.to_csv('customer_data.csv', index=False)
print("✅ 'customer_data.csv' successfully generated with 500 records.")
