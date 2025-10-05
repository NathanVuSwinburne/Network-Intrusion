# ============================================================================
# NETWORK INTRUSION DETECTION - DBSCAN CLUSTERING ANALYSIS
# ============================================================================

import pandas as pd
import numpy as np
import os
import glob
import pickle
import warnings
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

output_dir = 'unsupervised_learning/Clustering/results_dbscan'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("DBSCAN CLUSTERING ANALYSIS")
print("="*80)

# ============================================================================
# LOAD AND MERGE DATA
# ============================================================================
print("\nLoading and merging data...")
train_files = glob.glob('data/processed_data_binary/X_train_scaled_binary_class_*.csv')
test_files = glob.glob('data/processed_data_binary/X_test_scaled_binary_class_*.csv')
le_files = glob.glob('data/processed_data_binary/label_encoder_binary_class_*.pkl')

if not train_files or not test_files:
    print("Error: Data files not found.")
    exit()

latest_train = max(train_files, key=os.path.getctime)
latest_test = max(test_files, key=os.path.getctime)
latest_le = max(le_files, key=os.path.getctime)

print(f"Loading: {os.path.basename(latest_train)}")
print(f"Loading: {os.path.basename(latest_test)}")

train_data = pd.read_csv(latest_train)
test_data = pd.read_csv(latest_test)

with open(latest_le, 'rb') as f:
    le = pickle.load(f)

# Merge datasets
merged_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
X_merged = merged_data.drop(columns=['label'])
y_merged = merged_data['label']

print(f"\nMerged dataset: {merged_data.shape}")
print(f"Benign: {(y_merged == 0).sum():,} ({(y_merged == 0).sum() / len(y_merged) * 100:.1f}%)")
print(f"Attack: {(y_merged == 1).sum():,} ({(y_merged == 1).sum() / len(y_merged) * 100:.1f}%)")

# Sample for faster clustering
SAMPLE_SIZE = 20000
print(f"\nSampling {SAMPLE_SIZE:,} samples for clustering...")
sample_idx = np.random.choice(len(X_merged), SAMPLE_SIZE, replace=False)
X_sample = X_merged.iloc[sample_idx]
y_sample = y_merged.iloc[sample_idx]

print(f"Sample - Benign: {(y_sample == 0).sum():,}, Attack: {(y_sample == 1).sum():,}")

# ============================================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================================
print("\n" + "="*80)
print("PCA DIMENSIONALITY REDUCTION")
print("="*80)

print("Applying PCA (2D)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sample)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")

# PCA for clustering (higher dimensions)
print("\nApplying PCA for clustering (10 components)...")
pca_cluster = PCA(n_components=10, random_state=42)
X_pca_cluster = pca_cluster.fit_transform(X_sample)

print("\nDetecting and removing outliers...")
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso.fit_predict(X_pca_cluster)
mask = outlier_labels == 1  # Keep only inliers

X_pca_cluster_clean = X_pca_cluster[mask]
X_sample_clean = X_sample[mask]
X_pca_clean = X_pca[mask]
y_sample_clean = y_sample[mask]

print(f"Removed {(~mask).sum()} outliers, kept {mask.sum()} samples")
print(f"Clustering PCA explained variance: {pca_cluster.explained_variance_ratio_.sum():.2%}")

# ============================================================================
# DETERMINE OPTIMAL EPS USING K-DISTANCE GRAPH
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL EPS PARAMETER")
print("="*80)

print("\nCalculating k-distance graph (k=4)...")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_pca_cluster_clean)
distances, indices = neighbors_fit.kneighbors(X_pca_cluster_clean)

# Sort distances
distances = np.sort(distances[:, 4], axis=0)

# Find elbow point (simplified approach)
eps_candidates = [0.3, 0.5, 0.7, 1.0, 1.5]
print(f"\nTesting eps values: {eps_candidates}")

# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("DBSCAN CLUSTERING")
print("="*80)

print("\nTesting different eps and min_samples combinations...")
best_score = -1
best_params = None
best_labels = None
results = []

for eps in eps_candidates:
    for min_samples in [5, 10, 15]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca_cluster_clean)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only calculate scores if we have valid clusters
        if n_clusters > 1 and n_noise < len(labels) * 0.5:
            # Filter out noise points for scoring
            mask_valid = labels != -1
            if mask_valid.sum() > 0:
                try:
                    sil_score = silhouette_score(X_pca_cluster_clean[mask_valid], labels[mask_valid])
                    db_score = davies_bouldin_score(X_pca_cluster_clean[mask_valid], labels[mask_valid])
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': sil_score,
                        'davies_bouldin': db_score
                    })
                    
                    print(f"eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, "
                          f"Noise={n_noise} ({n_noise/len(labels)*100:.1f}%), "
                          f"Silhouette={sil_score:.4f}, DB={db_score:.4f}")
                    
                    if sil_score > best_score:
                        best_score = sil_score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels.copy()
                except:
                    print(f"eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, "
                          f"Noise={n_noise} - Could not calculate scores")
        else:
            print(f"eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, "
                  f"Noise={n_noise} ({n_noise/len(labels)*100:.1f}%) - Invalid clustering")

if best_params is None:
    print("\nNo valid clustering found. Using default parameters...")
    best_params = {'eps': 0.5, 'min_samples': 10}
    dbscan_final = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    best_labels = dbscan_final.fit_predict(X_pca_cluster_clean)
    best_score = -1

print(f"\n✓ Best parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
if best_score > 0:
    print(f"  Silhouette Score: {best_score:.4f}")

# Analyze final clustering
cluster_labels = best_labels
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\nFinal DBSCAN Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

print("\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    if cluster == -1:
        print(f"  Noise: {count:,} ({count/len(cluster_labels)*100:.1f}%)")
    else:
        print(f"  Cluster {cluster}: {count:,} ({count/len(cluster_labels)*100:.1f}%)")

# Analyze cluster composition
print("\nCluster Composition Analysis:")
print("-" * 60)
cluster_types = {}
for cluster in unique:
    cluster_mask = cluster_labels == cluster
    cluster_true_labels = y_sample_clean[cluster_mask]
    
    benign_count = (cluster_true_labels == 0).sum()
    attack_count = (cluster_true_labels == 1).sum()
    total = len(cluster_true_labels)
    
    benign_pct = benign_count / total * 100
    attack_pct = attack_count / total * 100
    
    # Determine cluster type
    if benign_pct > 70:
        cluster_type = "Mostly Benign"
    elif attack_pct > 70:
        cluster_type = "Mostly Attack"
    else:
        cluster_type = "Mixed"
    
    cluster_types[cluster] = cluster_type
    
    label = "Noise" if cluster == -1 else f"Cluster {cluster}"
    print(f"\n{label} ({total:,} samples) - {cluster_type.upper()}")
    print(f"  Benign: {benign_count:,} ({benign_pct:.1f}%)")
    print(f"  Attack: {attack_count:,} ({attack_pct:.1f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: PCA with True Labels
print("\n1. PCA - True Labels...")
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca_clean[:, 0], X_pca_clean[:, 1], c=y_sample_clean, cmap='coolwarm', alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('PCA Visualization - True Labels (DBSCAN)', fontsize=14, fontweight='bold')
legend_elements = [
    Patch(facecolor='#3b4cc0', label='Benign (0)', alpha=0.6),
    Patch(facecolor='#b40426', label='Attack (1)', alpha=0.6)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{output_dir}/dbscan_1_pca_true_labels.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: dbscan_1_pca_true_labels.png")

# Plot 2: PCA with DBSCAN Clusters
print("2. PCA - DBSCAN Clusters...")
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca_clean[:, 0], X_pca_clean[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title(f'PCA Visualization - DBSCAN Clusters (eps={best_params["eps"]}, min_samples={best_params["min_samples"]})', 
             fontsize=14, fontweight='bold')
# Create legend
n_colors = len(unique)
colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
legend_elements = []
for i, cluster in enumerate(unique):
    label = f'Noise: {cluster_types[cluster]}' if cluster == -1 else f'C{cluster}: {cluster_types[cluster]}'
    legend_elements.append(Patch(facecolor=colors[i], label=label, alpha=0.6))
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{output_dir}/dbscan_2_pca_dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: dbscan_2_pca_dbscan_clusters.png")

# Plot 3: K-Distance Graph
print("3. K-Distance Graph...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(distances, linewidth=2)
ax.axhline(y=best_params['eps'], color='red', linestyle='--', linewidth=2, label=f'Selected eps={best_params["eps"]}')
ax.set_xlabel('Data Points (sorted by distance)', fontsize=12)
ax.set_ylabel('4-th Nearest Neighbor Distance', fontsize=12)
ax.set_title('K-Distance Graph for DBSCAN eps Selection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/dbscan_3_k_distance_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: dbscan_3_k_distance_graph.png")

# Plot 4: Cluster Distribution Comparison
print("4. Cluster Distribution Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# True labels
unique_true, counts_true = np.unique(y_sample_clean, return_counts=True)
bars1 = ax1.bar(['Benign', 'Attack'], counts_true, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('True Label Distribution', fontsize=13, fontweight='bold')
for i, (bar, count) in enumerate(zip(bars1, counts_true)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count:,}\n({count/len(y_sample_clean)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# DBSCAN clusters
unique_clust, counts_clust = np.unique(cluster_labels, return_counts=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clust)))
labels = ['Noise' if c == -1 else f'Cluster {c}' for c in unique_clust]
bars2 = ax2.bar(labels, counts_clust, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(f'DBSCAN Cluster Distribution ({n_clusters} clusters)', fontsize=13, fontweight='bold')
for i, (bar, count) in enumerate(zip(bars2, counts_clust)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count:,}\n({count/len(cluster_labels)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/dbscan_4_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: dbscan_4_cluster_distribution.png")

# Plot 5: Parameter Comparison (if we have results)
if len(results) > 0:
    print("5. Parameter Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by eps
    eps_values = sorted(set([r['eps'] for r in results]))
    for eps_val in eps_values:
        eps_results = [r for r in results if r['eps'] == eps_val]
        min_samples_vals = [r['min_samples'] for r in eps_results]
        sil_scores = [r['silhouette'] for r in eps_results]
        ax1.plot(min_samples_vals, sil_scores, 'o-', label=f'eps={eps_val}', linewidth=2, markersize=8)
    
    ax1.set_xlabel('min_samples', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('DBSCAN Parameter Tuning - Silhouette Score', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Number of clusters
    for eps_val in eps_values:
        eps_results = [r for r in results if r['eps'] == eps_val]
        min_samples_vals = [r['min_samples'] for r in eps_results]
        n_clusters_vals = [r['n_clusters'] for r in eps_results]
        ax2.plot(min_samples_vals, n_clusters_vals, 'o-', label=f'eps={eps_val}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('min_samples', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('DBSCAN Parameter Tuning - Number of Clusters', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dbscan_5_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: dbscan_5_parameter_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save detailed results
results_file = f'{output_dir}/dbscan_clustering_results.txt'
with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DBSCAN CLUSTERING ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Samples: {len(X_merged):,}\n")
    f.write(f"Samples Used: {len(X_sample):,}\n")
    f.write(f"Features: {X_sample.shape[1]}\n\n")
    
    f.write("="*80 + "\n")
    f.write("DATASET STATISTICS\n")
    f.write("="*80 + "\n")
    f.write(f"Training set: {len(train_data):,}\n")
    f.write(f"Test set: {len(test_data):,}\n")
    f.write(f"Merged set: {len(merged_data):,}\n\n")
    f.write(f"Benign (0): {(y_merged == 0).sum():,} ({(y_merged == 0).sum() / len(y_merged) * 100:.2f}%)\n")
    f.write(f"Attack (1): {(y_merged == 1).sum():,} ({(y_merged == 1).sum() / len(y_merged) * 100:.2f}%)\n\n")
    
    f.write("="*80 + "\n")
    f.write("DBSCAN CLUSTERING RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Best eps: {best_params['eps']}\n")
    f.write(f"Best min_samples: {best_params['min_samples']}\n")
    if best_score > 0:
        f.write(f"Silhouette Score: {best_score:.4f}\n")
    f.write(f"Number of clusters: {n_clusters}\n")
    f.write(f"Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)\n\n")
    
    f.write("Cluster Distribution:\n")
    for cluster, count in zip(unique, counts):
        if cluster == -1:
            f.write(f"  Noise: {count:,} ({count/len(cluster_labels)*100:.2f}%)\n")
        else:
            f.write(f"  Cluster {cluster}: {count:,} ({count/len(cluster_labels)*100:.2f}%)\n")
    
    if len(results) > 0:
        f.write("\nAll Parameter Combinations Tested:\n")
        for result in results:
            f.write(f"  eps={result['eps']}, min_samples={result['min_samples']}: "
                    f"Clusters={result['n_clusters']}, Noise={result['n_noise']}, "
                    f"Silhouette={result['silhouette']:.4f}, Davies-Bouldin={result['davies_bouldin']:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("PCA RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Explained Variance (2D): {pca.explained_variance_ratio_.sum():.4f}\n")
    f.write(f"  Component 1: {pca.explained_variance_ratio_[0]:.4f}\n")
    f.write(f"  Component 2: {pca.explained_variance_ratio_[1]:.4f}\n")

print(f"Results saved: {results_file}")

# Save cluster labels
labels_df = pd.DataFrame({
    'cluster': cluster_labels,
    'true_label': y_sample_clean.values
})
labels_file = f'{output_dir}/dbscan_cluster_labels.csv'
labels_df.to_csv(labels_file, index=False)
print(f"Labels saved: {labels_file}")

# Save model parameters
model_file = f'{output_dir}/dbscan_params.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({'best_params': best_params, 'pca': pca, 'cluster_labels': cluster_labels}, f)
print(f"Model parameters saved: {model_file}")

print("\n" + "="*80)
print("✓ DBSCAN CLUSTERING ANALYSIS COMPLETED!")
print("="*80)
print(f"\nResults directory: {output_dir}/")
print("\nGenerated files:")
print("  • dbscan_clustering_results.txt")
print("  • dbscan_cluster_labels.csv")
print("  • dbscan_params.pkl")
print("  • 4-5 visualization plots (PNG with 'dbscan_' prefix)")
if best_score > 0:
    print(f"\nBest DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}, Silhouette={best_score:.4f}")
else:
    print(f"\nBest DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
print(f"Clusters found: {n_clusters}, Noise: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
print("="*80)