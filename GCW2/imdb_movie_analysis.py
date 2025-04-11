#%%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch

# Set up plotting and saving configurations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
SAVE_FIGURES = 1
os.makedirs("./images", exist_ok=True)
os.makedirs("./datasets", exist_ok=True)

def maybe_savefig(filename):
    """
    Save figures to an images directory if SAVE_FIGURES is True/1.
    Else, do not save figures.

    Args:
    filename (str): Name of the file to save the figure as
    """
    if SAVE_FIGURES:
        plt.savefig(f"./images/{filename}")

class DataProcessor:
    """
    DataProcessor class to load, preprocess and standardise data

    Attributes:
    file_path (str): Path to the data file
    df (pd.DataFrame): Dataframe to store the data
    numeric_columns (list): List of numeric columns in the dataframe

    Methods:
    load_data(): Load the data from the file_path
    filter_numeric_columns(): Filter the dataframe to only include numeric columns
    rename_column(old_column_name, new_column_name): Rename a column in the dataframe
    standardise(): Standardise the numeric columns in the dataframe
    """
    def __init__(self, file_path):
        # Set up file path and data frame
        self.file_path = file_path
        self.df = None
        self.numeric_columns = None

    def load_data(self):
        # Load data from file path
        self.df = pd.read_csv(self.file_path)

    def filter_numeric_columns(self):
        # Filter dataframe for numeric columns
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.drop('Year', errors='ignore')
        self.df = self.df[self.numeric_columns]

    def remove_rows_with_missing_values(self):
        # Remove rows with missing values to ensure PCA works
        self.df.dropna(axis=0, inplace=True)
        return self.df

    def rename_column(self, old_column_name, new_column_name):
        # Rename a column in the dataframe
        self.df.rename(columns={old_column_name: new_column_name}, inplace=True)
        return self.df

    def standardise(self):
        # Standardise columns in dataframe to prevent bias to majority columns
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(self.df)
        self.df = pd.DataFrame(scaled_array, columns=self.df.columns)
        return self.df

    def get_aligned_original_data(self):
        # Reload original dataset and align to filtered numeric data
        original = pd.read_csv(self.file_path)
        original.rename(columns={'Unnamed: 0': 'Ranking'}, inplace=True)
        return original.loc[self.df.index].reset_index(drop=True)

class EDASummary:
    """
    EDASummary class to perform summary-level exploratory data analysis

    Attributes:
    df (pd.DataFrame): Dataframe to store the data

    Methods:
    display_shape(): Display the shape of the data
    display_data_types(): Display the data types of the columns
    display_summary_statistics(): Display the summary statistics of the data
    display_missing_values(): Display the missing values in the data
    """
    def __init__(self, df):
        # Set up dataframe
        self.df = df

    def display_shape(self):
        # Display data shape
        print('Data Shape:')
        print(self.df.shape)

    def display_data_types(self):
        # Display data types
        print('Data Types:')
        print(self.df.dtypes)

    def display_summary_statistics(self):
        # Display summary statistics
        print('========== Summary Stats ==========' )
        print(self.df.describe())

    def display_missing_values(self):
        # Display missing values
        print('Missing Values:')
        print(self.df.isnull().sum())

class EDAVisuals:
    """
    EDAVisuals class to perform visual exploratory data analysis

    Attributes:
    df (pd.DataFrame): Dataframe to store the data

    Methods:
    display_heatmap(): Display the heatmap of the correlation matrix
    display_distribution(): Display the distribution of the data
    display_boxplot(): Display the boxplot of the data
    display_pairplot(): Display the pairplot of the data
    """
    def __init__(self, df):
        # Set up dataframe
        self.df = df

    def display_heatmap(self):
        # Display correlation matrix heatmap
        plt.figure(figsize=(10,10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', annot_kws={"size": 14})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        maybe_savefig("correlation_matrix.png")
        plt.show()

    def display_distribution(self):
        # Display data distribution
        self.df.hist(bins=20, figsize=(20,20))
        plt.suptitle('Distribution of Data')
        plt.tight_layout()
        maybe_savefig("distribution.png")
        plt.show()

    def display_boxplot(self):
        # Display boxplot of data
        self.df.boxplot(figsize=(20,10))
        plt.title('Boxplot of Data')
        plt.tight_layout()
        maybe_savefig("boxplot.png")
        plt.show()

    def display_pairplot(self):
        # Display pairplot of data
        sns.pairplot(self.df)
        plt.suptitle('Pairplot of Data')
        plt.tight_layout()
        maybe_savefig("pairplot.png")
        plt.show()

class PCAProcessor:
    """
    PCAProcessor class to perform Principal Component Analysis on the data

    Attributes:
    df (pd.DataFrame): Dataframe containing the standardised data
    pca (PCA): PCA object to perform the analysis
    principal_components (np.array): Array to store the principal components

    Methods:
    get_scores(): Get the principal component scores
    get_loadings(): Get the PCA loadings
    explained_variance(): Get the explained variance ratio
    get_n_components_for_variance(threshold): Get number of components to reach specified variance threshold
    get_aligned_original_data(): Reload original dataset and align to filtered numeric
    """
    def __init__(self, df, n_components=2):
        # Set up dataframe and PCA object
        self.df = df
        self.data = df.values
        self.index = df.index
        self.pca = PCA(n_components=n_components)
        self.principal_components = self.pca.fit_transform(self.data)

    def get_scores(self):
        # Get principal component scores
        return pd.DataFrame(
            data=self.principal_components, 
            index=self.index,
            columns=[f'PC{i+1}' for i in range(self.principal_components.shape[1])]
        )

    def get_loadings(self, pc_df):
        # Get PCA loadings
        return pd.DataFrame(
            data=self.pca.components_.T, 
            columns=[f"PC{i+1}" for i in range(pc_df.shape[1])], 
            index=self.df.columns
        )

    def explained_variance(self):
        # Get explained variance ratio
        return self.pca.explained_variance_ratio_

    def get_n_components_for_variance(self, threshold=0.95):
        # Get number of components required to reach the specified cumulative explained variance
        cum_var = np.cumsum(self.explained_variance())
        return np.argmax(cum_var >= threshold) + 1
    
class PCAVisualiser:
    """
    PCAVisualiser class to display PCA results visually

    Attributes:
    processor (PCAProcessor): PCAProcessor object containing PCA results

    Methods:
    display_scree_plot(): Display explained variance per component
    display_cumulative_variance(): Display cumulative explained variance
    display_biplot(): Display combined PCA score + loadings biplot
    """
    def __init__(self, processor: PCAProcessor):
        # Set up PCA processor from PCAProcessor class
        self.processor = processor

    def display_scree_plot(self):
        # Display scree plot to show explained variance per component
        var = self.processor.explained_variance()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(var) + 1), var, marker='o', linestyle='--')
        plt.title("Scree Plot: Explained Variance by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        plt.tight_layout()
        maybe_savefig("scree_plot.png")
        plt.show()

    def display_cumulative_variance(self):
        # Display cumulative explained variance to show it plateaus to 1, as expected
        cum_var = np.cumsum(self.processor.explained_variance())
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--')
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        maybe_savefig("cumulative_variance.png")
        plt.show()

    def display_biplot(self):
        # Display biplot of PCA scores and loadings
        scores = self.processor.principal_components
        coeff = self.processor.pca.components_.T
        labels = self.processor.df.columns

        xs = scores[:, 0]
        ys = scores[:, 1]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        plt.figure(figsize=(10, 7))
        plt.scatter(xs * scalex, ys * scaley, c='grey', alpha=0.5)
        for i in range(coeff.shape[0]):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', width=0.002, head_width=0.02)
            plt.text(coeff[i, 0] * 1.1, coeff[i, 1] * 1.1, labels[i], color='b', fontsize=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Biplot of Numeric Features")
        plt.axhline(y=0, color='black', linewidth=0.8)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.grid(False)
        plt.tight_layout()
        maybe_savefig("pca_biplot.png")
        plt.show()

class KMeansClusteringProcessor:
    """
    KMeansClusteringProcessor class to perform KMeans clustering on the data
    
    Attributes:
    data (np.array): Array containing the data
    max_k (int): Maximum number of clusters to test
    silhouette_scores (dict): Dictionary to store silhouette scores for each k
    optimal_k (int): Optimal number of clusters based on silhouette score
    labels (np.array): Array to store cluster labels
    
    Methods:
    find_optimal_k(): Find the optimal number of clusters based on silhouette score
    fit_final_model(): Fit the final KMeans model with optimal number of clusters
    """
    def __init__(self, data, max_k=10):
        # Set up data and parameters
        self.data = data
        self.max_k = max_k
        self.silhouette_scores = {}
        self.optimal_k = None
        self.labels = None

    def find_optimal_k(self):
        # Find optimal number of clusters based on silhouette score
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, cluster_labels)
            self.silhouette_scores[k] = score
            print(f"Silhouette Score for k={k}: {score:.3f}")
        self.optimal_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        print(f"\nOptimal number of clusters (based on silhouette score): {self.optimal_k}")

    def fit_final_model(self):
        # Fit final KMeans model with optimal number of clusters
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42)
        self.labels = kmeans.fit_predict(self.data)
        return self.labels

class KMeansClusteringVisualiser:
    """
    KMeansClusteringVisualiser class to display KMeans clustering results visually
    
    Attributes:
    processor (KMeansClusteringProcessor): KMeansClusteringProcessor object containing KMeans results
    
    Methods:
    plot_silhouette_scores(): Plot silhouette scores for different k
    plot_clusters(pca_data): Plot KMeans clusters on PCA components
    """
    def __init__(self, processor: KMeansClusteringProcessor):
        # Set up KMeans processor from KMeansClusteringProcessor
        self.processor = processor

    def plot_silhouette_scores(self):
        # Plot silhouette scores for different k
        plt.figure(figsize=(8, 5))
        plt.plot(list(self.processor.silhouette_scores.keys()), list(self.processor.silhouette_scores.values()), marker='o', linestyle='--')
        plt.title("Silhouette Scores for Different k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.tight_layout()
        maybe_savefig("kmeans_silhouette_scores.png")
        plt.show()

    def plot_clusters(self, pca_data):
        # Plot KMeans clusters on PCA components
        labels = self.processor.labels
        plt.figure(figsize=(8, 5))
        for cluster in np.unique(labels):
            ix = np.where(labels == cluster)
            plt.scatter(pca_data[ix, 0], pca_data[ix, 1], label=f"Cluster {cluster}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("KMeans Clusters Visualised on PCA Components")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        maybe_savefig("kmeans_clusters.png")
        plt.show()

class HierarchicalClusteringProcessor:
    """
    HierarchicalClusteringProcessor class to perform Hierarchical clustering on the data
    
    Attributes:
    data (np.array): Array containing the data
    linkage_matrix (np.array): Array to store the linkage matrix
    labels (np.array): Array to store cluster labels
    
    Methods:
    compute_linkage(method='ward'): Compute the linkage matrix using Ward's method
    assign_clusters(threshold_ratio=0.7): Assign clusters based on a threshold distance
    """
    def __init__(self, data):
        # Set up data and parameters
        self.data = data
        self.linkage_matrix = None
        self.labels = None

    def compute_linkage(self, method='ward'):
        # Compute linkage matrix using Ward's.
        self.linkage_matrix = sch.linkage(self.data, method=method)

    def assign_clusters(self, threshold_ratio=0.7):
        # Assign clusters based on a threshold distance
        # e.g. 70% of max distance
        max_d = np.max(self.linkage_matrix[:, 2]) * threshold_ratio
        self.labels = sch.fcluster(self.linkage_matrix, max_d, criterion='distance')
        print(f"\nNumber of hierarchical clusters: {len(np.unique(self.labels))}")
        return self.labels

class HierarchicalClusteringVisualiser:
    """
    HierarchicalClusteringVisualiser class to display Hierarchical clustering results visually
    
    Attributes:
    processor (HierarchicalClusteringProcessor): HierarchicalClusteringProcessor object containing Hierarchical clustering results
    
    Methods:
    plot_dendrogram(labels): Plot dendrogram for hierarchical clustering
    plot_clusters(pca_data): Plot Hierarchical clusters on PCA components
    """
    def __init__(self, processor: HierarchicalClusteringProcessor):
        # Set up Hierarchical processor from HierarchicalClusteringProcessor
        self.processor = processor

    def plot_dendrogram(self, labels):
        # Plot dendrogram for hierarchical clustering
        plt.figure(figsize=(10, 7))
        sch.dendrogram(self.processor.linkage_matrix, labels=labels, leaf_rotation=90)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.grid(False)
        maybe_savefig("hierarchical_dendrogram.png")
        plt.show()

    def plot_clusters(self, pca_data):
        # Plot Hierarchical clusters on PCA components
        labels = self.processor.labels
        plt.figure(figsize=(8, 5))
        for cluster in np.unique(labels):
            ix = np.where(labels == cluster)
            plt.scatter(pca_data[ix, 0], pca_data[ix, 1], label=f"Cluster {cluster}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Hierarchical Clustering Visualised on PCA Components")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        maybe_savefig("hierarchical_clusters.png")
        plt.show()

if __name__ == "__main__":
    # Data loading and preprocessing
    file_path = './datasets/IMDB-Movies.csv'
    data_processor = DataProcessor(file_path)
    data_processor.load_data()
    data_processor.filter_numeric_columns()
    data_processor.rename_column('Unnamed: 0', 'Ranking')
    data_processor.remove_rows_with_missing_values()
    df_std = data_processor.standardise()

    # Summary EDA
    summary = EDASummary(df_std)
    summary.display_shape()
    summary.display_data_types()
    summary.display_summary_statistics()
    summary.display_missing_values()

    # Visual EDA
    eda_viz = EDAVisuals(df_std)
    eda_viz.display_heatmap()
    eda_viz.display_distribution()
    eda_viz.display_boxplot()
    eda_viz.display_pairplot()

    # We must remove Ranking column as it is highly correlated with rating
    df_std = df_std.drop(columns=['Ranking'])

    # PCA (full)
    pca_runner = PCAProcessor(df_std, n_components=df_std.shape[1])
    pc_df = pca_runner.get_scores()
    loadings = pca_runner.get_loadings(pc_df)

    # Visualise PCA
    pca_viz = PCAVisualiser(pca_runner)
    pca_viz.display_scree_plot()
    pca_viz.display_cumulative_variance()
    pca_viz.display_biplot()

    # Determine how many components explain 95% variance
    n_components_95 = pca_runner.get_n_components_for_variance(threshold=0.95)
    print(f"Number of components to explain 95% variance: {n_components_95}")

    # Re-run PCA with reduced components
    pca_runner_reduced = PCAProcessor(df_std, n_components=n_components_95)
    reduced_pca_scores = pca_runner_reduced.get_scores().values

    # KMeans Clustering on reduced PCA
    kmeans_proc = KMeansClusteringProcessor(reduced_pca_scores)
    kmeans_proc.find_optimal_k()
    kmeans_labels = kmeans_proc.fit_final_model()
    df_std["KMeans Cluster"] = kmeans_labels

    # Visualise KMeans Clustering
    kmeans_viz = KMeansClusteringVisualiser(kmeans_proc)
    kmeans_viz.plot_silhouette_scores()
    kmeans_viz.plot_clusters(pca_runner.principal_components)

    # Print KMeans Cluster Summary
    print("\nKMeans Cluster Summary:\n")
    print(df_std.groupby("KMeans Cluster").mean())
    print("\nKMeans Cluster Counts:")
    print(df_std["KMeans Cluster"].value_counts())

    # Feature distributions by cluster
    for col in df_std.columns.difference(["KMeans Cluster", "Hierarchical Cluster"]):
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_std, x='KMeans Cluster', y=col)
        plt.title(f'{col} by KMeans Cluster')
        plt.tight_layout()
        maybe_savefig(f"{col}_kmeans_cluster_boxplot.png")
        plt.show()

    # Cluster Stability Check
    print("\nKMeans Cluster Stability Check:")
    seeds = [0, 21, 42, 99]
    for seed in seeds:
        kmeans_alt = KMeans(n_clusters=kmeans_proc.optimal_k, random_state=seed)
        alt_labels = kmeans_alt.fit_predict(reduced_pca_scores)
        ari = adjusted_rand_score(kmeans_labels, alt_labels)
        print(f"Adjusted Rand Index vs seed {seed}: {ari:.3f}")

    # Hierarchical Clustering
    hier_proc = HierarchicalClusteringProcessor(reduced_pca_scores)
    hier_proc.compute_linkage()
    hier_labels = hier_proc.assign_clusters()
    df_std["Hierarchical Cluster"] = hier_labels

    # Visualise Hierarchical Clustering
    hier_viz = HierarchicalClusteringVisualiser(hier_proc)
    hier_viz.plot_dendrogram(labels=df_std.index.tolist())
    hier_viz.plot_clusters(pca_runner.principal_components)

    # Print Hierarchical Cluster Summary
    print("\nHierarchical Cluster Summary:\n")
    print(df_std.groupby("Hierarchical Cluster").mean())
    print("\nHierarchical Cluster Counts:")
    print(df_std["Hierarchical Cluster"].value_counts())

    # Save final datasets for further analysis
    print('Analysis Completed!')
    print('Saving Final Results with Movie Titles...')

    # Get aligned original metadata from DataProcessor
    original_df = data_processor.get_aligned_original_data()

    # Safety checks
    assert len(original_df) == len(df_std), "Row count mismatch after preprocessing!"
    assert len(original_df) == len(pc_df), "Row count mismatch with PCA DataFrame!"

    # Define final outputs
    final_outputs = {
        "./datasets/movies_clustered.csv": original_df.assign(
            **{
                "KMeans Cluster": df_std["KMeans Cluster"].values,
                "Hierarchical Cluster": df_std["Hierarchical Cluster"].values,
            }
        ),
        "./datasets/movies_pca.csv": original_df.join(pc_df)
    }

    # Save each output
    for path, df in final_outputs.items():
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    # Confirm completion
    print('All final datasets saved successfully!')



# %%
