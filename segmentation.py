import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. Load Data
try:
    df = pd.read_csv('customer_data.csv')
except FileNotFoundError:
    print("❌ Error: 'customer_data.csv' not found. Run generate_data.py first.")
    exit()

# 2. Feature Selection & Scaling
# We use Income, Spending Score, and Age for behavioral/demographic segmentation
features = ['Age', 'Annual_Income_k', 'Spending_Score']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find Optimal Clusters using the Elbow Method
wcss = []
cluster_range = range(1, 11)
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Save the Elbow Plot to show project steps
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.savefig('elbow_plot.png')
print("📈 Elbow plot saved as 'elbow_plot.png'")

# 4. Apply K-Means with Optimal Clusters (Selected K=4 based on distribution)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters intuitively based on characteristics
cluster_map = {0: "Budget-Conscious Seniors", 1: "High-Income Loyalists", 2: "Mid-Range Average Users", 3: "Young High-Spenders"}
df['Segment_Name'] = df['Cluster'].map(cluster_map)

# 5. Profile Clusters (Generate insights)
print("\n📋 CUSTOMER SEGMENT PROFILES (AVERAGES):")
profile = df.groupby('Segment_Name')[features].mean().round(1)
print(profile)

# 6. Interactive 3D Segment Visualization
fig = px.scatter_3d(
    df, x='Age', y='Annual_Income_k', z='Spending_Score',
    color='Segment_Name',
    title='Interactive Customer Segments (3D K-Means Clustering)',
    labels={'Annual_Income_k': 'Annual Income ($k)', 'Spending_Score': 'Spending Score (1-100)'},
    opacity=0.8
)
fig.show()
