import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data_path = r'C:\Users\nandi\OneDrive\Desktop\oasis\task2\ifood_df.csv'
data = pd.read_csv(data_path)
print("Data preview:\n", data.head())
print("Missing values:\n", data.isnull().sum())
print("Descriptive Statistics:\n", data.describe())
features = data[['Income', 'Kidhome', 'Teenhome', 'Recency', 
                 'MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
features = features.fillna(features.mean())
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(scaled_features)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Income'], y=data['MntWines'], hue=data['Cluster'], palette='viridis')
plt.title('Customer Segmentation Based on Income and Wine Purchases')
plt.xlabel('Income')
plt.ylabel('Amount Spent on Wine')
plt.show()
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Customers per Cluster:\n", data['Cluster'].value_counts())
