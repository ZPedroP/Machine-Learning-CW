# === Imports ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
import itertools

# === Load the dataset ===
file_path = 'datasets/IMDB-Movies.csv'
df = pd.read_csv(file_path)

# Display basic info and first few rows to inspect column names and types
print(df.info())
print(df.head())

# === Rename for clarity ===
df = df.rename(columns={"Unnamed: 0": "Ranking"})

# Select numerical columns (excluding Year)
numerical_df = df[["Ranking", "Runtime..Minutes.", "Rating",
                    "Votes", "Revenue..Millions."]]

# Compute correlation matrix
correlation_matrix = numerical_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

# Show correlations with Rating
print("Correlation with Rating:")
print(correlation_matrix["Rating"].sort_values(ascending=False))

# === Imputation and Standardization ===

# Drop Ranking and Year, and prepare numerical features
features = ["Runtime..Minutes.", "Rating", "Votes", "Revenue..Millions."]
X = df[features].copy()

# Impute missing values in Revenue using median (robust to skew)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Convert back to DataFrame for reference
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("Scaled Numerical Features (First 5 Rows):")
print(X_scaled_df.head())

# === Perform PCA ===
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Component loadings
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' 
                                 for i in range(len(features))], index=features)

print("Explained Variance Ratio:")
print(explained_variance_ratio)

print("Cumulative Explained Variance:")
print(cumulative_variance)

print("PCA Component Loadings:")
print(loadings)

# 2D PCA projection
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', edgecolor='k', s=60)
plt.xlabel("Principal Component 1 (44.3%)")
plt.ylabel("Principal Component 2 (28.8%)")
plt.title("2D PCA Projection of IMDB Movies (Numerical Features)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === KMeans Clustering ===

# Try different values of k for K-Means
k_values = range(2, 8)
silhouette_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot silhouette and Davies-Bouldin scores
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(k_values, silhouette_scores, 'bo-', label='Silhouette Score')
ax2.plot(k_values, db_scores, 'rs-', label='Davies-Bouldin Index')

ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score", color='b')
ax2.set_ylabel("Davies-Bouldin Index", color='r')
plt.title("K-Means Clustering Evaluation")
fig.tight_layout()
plt.show()

print("Silhouette Scores by k:")
print(silhouette_scores)

print("Davies-Bouldin Scores by k:")
print(db_scores)

# Run KMeans with k=2 and k=5
kmeans_2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_scaled)
kmeans_5 = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled)

# Add cluster labels to PCA data for plotting
df["Cluster_k2"] = kmeans_2.labels_
df["Cluster_k5"] = kmeans_5.labels_
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# Plot PCA with k=2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster_k2", palette="Set1", s=80)
plt.title("PCA Projection with K-Means Clustering (k=2)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot PCA with k=5
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster_k5", palette="Set2", s=80)
plt.title("PCA Projection with K-Means Clustering (k=5)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Group summary statistics by cluster
cluster_summary_k2 = df.groupby("Cluster_k2")[features].mean().round(2)
cluster_summary_k5 = df.groupby("Cluster_k5")[features].mean().round(2)

print("KMeans k=2 Cluster Summary:")
print(cluster_summary_k2)

print("KMeans k=5 Cluster Summary:")
print(cluster_summary_k5)

# Expand genre into multiple dummy columns for analysis
genre_split = df['Genre'].str.get_dummies(sep=',')

# Combine with cluster labels
genre_cluster_df = pd.concat([df[["Cluster_k2", "Cluster_k5"]], genre_split], axis=1)

# Average genre presence per cluster (as proportions)
genre_by_cluster_k2 = genre_cluster_df.groupby("Cluster_k2").mean().round(2).T
genre_by_cluster_k5 = genre_cluster_df.groupby("Cluster_k5").mean().round(2).T

print("Genre Distribution by k=2 Cluster:")
print(genre_by_cluster_k2)

print("Genre Distribution by k=5 Cluster:")
print(genre_by_cluster_k5)

# Inspect top recurring directors
top_directors = df['Director'].value_counts()
top_directors = top_directors[top_directors > 1]

# Map directors with 2+ movies
df['Top_Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')

# Cross-tabulation of Top Directors vs Clusters
director_cluster_k2 = pd.crosstab(df['Top_Director'], df['Cluster_k2'])
director_cluster_k5 = pd.crosstab(df['Top_Director'], df['Cluster_k5'])

print("Top Directors vs k=2 Clusters:")
print(director_cluster_k2)

print("Top Directors vs k=5 Clusters:")
print(director_cluster_k5)

# === Actor Analysis ===

# Extract and flatten actor lists
df['Actor_List'] = df['Actors'].str.split(',').apply(lambda x: [a.strip() for a in x])
all_actors = list(itertools.chain.from_iterable(df['Actor_List']))
actor_counts = Counter(all_actors)

# Focus on frequently occurring actors (in 2+ movies)
top_actors = {actor for actor, count in actor_counts.items() if count > 1}

# Create binary matrix of actors
for actor in top_actors:
    df[actor] = df['Actor_List'].apply(lambda x: int(actor in x))

# Aggregate actor presence by cluster (k=5)
actor_k5= df.groupby("Cluster_k5")[[actor for actor in top_actors]].sum().astype(int).T

print("Top Actor Occurrences by k=5 Cluster:")
print(actor_k5)
