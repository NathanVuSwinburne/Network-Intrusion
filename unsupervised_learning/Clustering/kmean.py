# ============================================================================
# NETWORK INTRUSION DETECTION - CLUSTERING ANALYSIS
# ============================================================================

import pandas as pd
import numpy as np
import os
import glob
import pickle
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

output_dir = 'unsupervised_learning/Clustering/results_kmean'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("CLUSTERING ANALYSIS - K-MEANS")
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
iso = IsolationForest(contamination=0.01, random_state=42)  # Remove top 0.1% outliers
outlier_labels = iso.fit_predict(X_pca_cluster)
mask = outlier_labels == 1  # Keep only inliers

X_pca_cluster_clean = X_pca_cluster[mask]
X_sample_clean = X_sample[mask]
X_pca_clean = X_pca[mask]
y_sample_clean = y_sample[mask]

print(f"Removed {(~mask).sum()} outliers, kept {mask.sum()} samples")
print(f"Clustering PCA explained variance: {pca_cluster.explained_variance_ratio_.sum():.2%}")
# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

print("\nTesting K values from 2 to 6...")
k_range = range(2, 7)
scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca_clean)
    sil_score = silhouette_score(X_pca_clean, labels)
    db_score = davies_bouldin_score(X_pca_clean, labels)
    
    scores.append({
        'k': k,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'inertia': kmeans.inertia_
    })
    
    print(f"K={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}, Inertia={kmeans.inertia_:.0f}")

# Select best K
best_idx = np.argmax([s['silhouette'] for s in scores])
best_k = scores[best_idx]['k']
print(f"\n✓ Best K: {best_k} (Silhouette: {scores[best_idx]['silhouette']:.4f})")

# Fit final model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_pca_clean)

print("\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count:,} ({count/len(cluster_labels)*100:.1f}%)")
# Analyze cluster composition and store cluster types
print("\nCluster Composition Analysis:")
print("-" * 60)
cluster_types = {}  # Store cluster type for legend
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
    
    cluster_types[cluster] = cluster_type  # Save for legend
    
    print(f"\nCluster {cluster} ({total:,} samples) - {cluster_type.upper()}")
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
ax.set_title('PCA Visualization - True Labels', fontsize=14, fontweight='bold')
# Box legend instead of colorbar
legend_elements = [
    Patch(facecolor='#3b4cc0', label='Benign (0)', alpha=0.6),
    Patch(facecolor='#b40426', label='Attack (1)', alpha=0.6)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_pca_true_labels.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: 1_pca_true_labels.png")

# Plot 2: PCA with K-Means Clusters
print("2. PCA - K-Means Clusters...")
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca_clean[:, 0], X_pca_clean[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title(f'PCA Visualization - K-Means Clusters (K={best_k})', fontsize=14, fontweight='bold')
# Box legend with cluster types
colors = plt.cm.viridis(np.linspace(0, 1, best_k))
legend_elements = [Patch(facecolor=colors[i], label=f'C{i}: {cluster_types[i]}', alpha=0.6) for i in range(best_k)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_pca_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: 2_pca_kmeans_clusters.png")

# Plot 3: Elbow Plots
print("3. K-Means Elbow Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Inertia
ax1.plot([s['k'] for s in scores], [s['inertia'] for s in scores], 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Plot - Inertia', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Silhouette
ax2.plot([s['k'] for s in scores], [s['silhouette'] for s in scores], 'go-', linewidth=2, markersize=8)
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Elbow Plot - Silhouette Score', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/3_elbow_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: 3_elbow_plots.png")

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

# K-Means clusters
unique_clust, counts_clust = np.unique(cluster_labels, return_counts=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clust)))
bars2 = ax2.bar([f'Cluster {i}' for i in unique_clust], counts_clust, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(f'K-Means Cluster Distribution (K={best_k})', fontsize=13, fontweight='bold')
for i, (bar, count) in enumerate(zip(bars2, counts_clust)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count:,}\n({count/len(cluster_labels)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/4_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: 4_cluster_distribution.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save detailed results
results_file = f'{output_dir}/clustering_results.txt'
with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SIMPLIFIED K-MEANS CLUSTERING ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")
    # f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    f.write("K-MEANS CLUSTERING RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Best K: {best_k}\n")
    f.write(f"Silhouette Score: {scores[best_idx]['silhouette']:.4f}\n")
    f.write(f"Davies-Bouldin Score: {scores[best_idx]['davies_bouldin']:.4f}\n")
    f.write(f"Inertia: {scores[best_idx]['inertia']:.2f}\n\n")
    
    f.write("Cluster Distribution:\n")
    for cluster, count in zip(unique, counts):
        f.write(f"  Cluster {cluster}: {count:,} ({count/len(cluster_labels)*100:.2f}%)\n")
    
    f.write("\nAll K Values Tested:\n")
    for score in scores:
        f.write(f"  K={score['k']}: Silhouette={score['silhouette']:.4f}, "
                f"Davies-Bouldin={score['davies_bouldin']:.4f}, Inertia={score['inertia']:.0f}\n")
    
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
labels_file = f'{output_dir}/cluster_labels.csv'
labels_df.to_csv(labels_file, index=False)
print(f"Labels saved: {labels_file}")

# Save model
model_file = f'{output_dir}/kmeans_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({'kmeans': kmeans_final, 'pca': pca}, f)
print(f"Model saved: {model_file}")

print("\n" + "="*80)
print("✓ CLUSTERING ANALYSIS COMPLETED!")
print("="*80)
print(f"\nResults directory: {output_dir}/")
print("\nGenerated files:")
print("  • clustering_results.txt")
print("  • cluster_labels.csv")
print("  • kmeans_model.pkl")
print("  • 4 visualization plots (PNG)")
print(f"\nBest K-Means: K={best_k}, Silhouette={scores[best_idx]['silhouette']:.4f}")
print("="*80)