"""
K-Means Clustering Tool - A Reusable Module
============================================
This module provides generic K-Means clustering functions that work with any dataset.

Usage:
    from modules.tool_kmeans import run_kmeans, find_optimal_k
    
    cluster_labels, centers, model = run_kmeans(df, target_column='Label', n_clusters=3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def run_kmeans(df, target_column, n_clusters=3, visualize=True, save_path=None):
    """
    Apply K-Means clustering to any dataframe.
    
    This function automatically:
    - Drops the specified target column before clustering
    - Applies StandardScaler to normalize all features
    - Fits K-Means and returns cluster assignments
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing features and target column.
    target_column : str
        Name of the target column to be dropped before clustering.
    n_clusters : int, default=3
        Number of clusters for K-Means algorithm.
    visualize : bool, default=True
        Whether to display a scatter plot of the clusters.
    save_path : str, optional
        Directory path to save visualization. If None, saves to current directory.
    
    Returns
    -------
    tuple
        (cluster_labels, cluster_centers, kmeans_model)
    
    Example
    -------
    >>> labels, centers, model = run_kmeans(df, 'PassFail', n_clusters=3)
    """
    
    # Step A: Separate features (X) from target (y) - Drop the target column
    print(f"\n{'='*60}")
    print("K-MEANS CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Store target for later comparison
    y = df[target_column].copy()
    
    # Drop target column to get features only
    X = df.drop(columns=[target_column])
    print(f"\n✓ Dropped target column: '{target_column}'")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Number of samples: {X.shape[0]}")
    
    # Step B: Apply StandardScaler to features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Applied StandardScaler to normalize features")
    
    # Step C: Initialize KMeans with n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Step D: Fit the model on scaled data
    cluster_labels = kmeans.fit_predict(X_scaled)
    print(f"✓ K-Means fitted with {n_clusters} clusters")
    
    # Step E: Print results and visualize
    print(f"\n{'='*60}")
    print("CLUSTERING RESULTS")
    print(f"{'='*60}")
    
    # Print cluster distribution
    print("\nCluster Distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # Print cluster centers
    print("\nCluster Centers (Scaled Values):")
    centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X.columns,
        index=[f"Cluster {i}" for i in range(n_clusters)]
    )
    print(centers_df.round(3).to_string())
    
    # Inertia (within-cluster sum of squares)
    print(f"\nInertia (WCSS): {kmeans.inertia_:.2f}")
    
    # Cross-tabulation with original target
    print(f"\n{'='*60}")
    print("CLUSTER vs TARGET COMPARISON")
    print(f"{'='*60}")
    crosstab = pd.crosstab(cluster_labels, y, margins=True)
    crosstab.index = [f"Cluster {i}" if i != 'All' else 'All' for i in crosstab.index]
    crosstab.columns.name = target_column
    print(crosstab)
    
    # Visualization
    if visualize:
        _visualize_clusters(X_scaled, cluster_labels, kmeans, y, target_column, n_clusters, save_path)
    
    return cluster_labels, kmeans.cluster_centers_, kmeans


def _visualize_clusters(X_scaled, cluster_labels, kmeans, y, target_column, n_clusters, save_path=None):
    """Internal function to create cluster visualizations using PCA."""
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    
    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.legend()
    
    # Plot 2: Original Labels for comparison
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter2, label=target_column)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'Original Labels ({target_column})')
    
    plt.tight_layout()
    
    # Save figure
    filename = 'kmeans_results.png'
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, filename)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    plt.show()


def find_optimal_k(df, target_column, k_range=range(2, 11), save_path=None):
    """
    Find optimal number of clusters using the Elbow Method.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    target_column : str
        Name of the target column to be dropped.
    k_range : range, default=range(2, 11)
        Range of k values to test.
    save_path : str, optional
        Directory path to save the elbow plot.
    
    Returns
    -------
    dict
        Dictionary with k values and their inertia scores.
    
    Example
    -------
    >>> elbow_results = find_optimal_k(df, 'PassFail', k_range=range(2, 8))
    """
    
    X = df.drop(columns=[target_column])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = 'elbow_method.png'
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, filename)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Elbow plot saved as '{filename}'")
    plt.show()
    
    return dict(zip(k_range, inertias))


if __name__ == "__main__":
    print("="*60)
    print("K-MEANS CLUSTERING TOOL")
    print("="*60)
    print("\nThis module provides reusable K-Means clustering functions.")
    print("\nAvailable functions:")
    print("  - run_kmeans(df, target_column, n_clusters=3)")
    print("  - find_optimal_k(df, target_column, k_range=range(2,11))")
    print("\nUsage:")
    print("  from modules.tool_kmeans import run_kmeans, find_optimal_k")
