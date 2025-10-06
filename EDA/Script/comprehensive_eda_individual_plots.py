"""
Comprehensive EDA for Network Intrusion Detection - Individual Plots
Author: Data Scientist - Cyber Security Domain
Date: 2025-10-06

This script performs EDA with selected charts saved separately:
1. Data Quality & Structure
2. Binary Classification Analysis (Attack vs Benign) 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ComprehensiveNetworkEDA:
    def __init__(self, file_path, output_dir='EDA/Outputs'):
        """Initialize the comprehensive EDA class"""
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None
        self.binary_labels = None
        self.chart_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load the merged dataset"""
        print("Loading merged dataset...")
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            
            # Create binary labels (BENIGN vs ATTACK)
            self.binary_labels = self.df['label'].apply(
                lambda x: 0 if x == 'BENIGN' else 1
            )
            self.df['binary_label'] = self.binary_labels
            
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def _save_plot(self, filename):
        """Helper to save and close plot"""
        filepath = f'{self.output_dir}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.chart_count += 1
        print(f"  [{self.chart_count}] ✓ {filename}")
    
    # ========================================================================
    # 🧩 SECTION 1: DATA QUALITY & STRUCTURE
    # ========================================================================
    
    def plot_1_dataset_sources(self):
        """Record count per source dataset"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'dataset_id' in self.df.columns:
            dataset_counts = self.df['dataset_id'].value_counts()
            dataset_names = {0: 'CIC-IDS-2017', 1: 'UNSW-NB15'}
            dataset_counts.index = dataset_counts.index.map(dataset_names)
            bars = ax.bar(dataset_counts.index, dataset_counts.values, 
                         color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
            ax.set_title('Record Count per Source Dataset', fontsize=16, fontweight='bold')
            ax.set_ylabel('Count', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        self._save_plot('1_01_dataset_sources.png')
    
    def plot_1_missing_values_bar(self):
        """Missing values per feature"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        missing_counts = self.df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            missing_counts.head(20).plot(kind='barh', ax=ax, color='coral', edgecolor='black')
            ax.set_title('Top 20 Features with Missing Values', fontsize=16, fontweight='bold')
            ax.set_xlabel('Missing Count', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Missing Values in Dataset', 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.axis('off')
        
        self._save_plot('1_02_missing_values.png')
    
    def plot_1_feature_types(self):
        """Feature type distribution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        type_counts = self.df.dtypes.value_counts()
        type_names = {
            'int64': 'Integer',
            'float64': 'Float',
            'object': 'Categorical',
            'bool': 'Boolean'
        }
        type_counts.index = type_counts.index.astype(str).map(
            lambda x: type_names.get(x, x)
        )
        colors_types = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax.pie(type_counts.values, 
                                           labels=type_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors_types[:len(type_counts)],
                                           startangle=90,
                                           explode=[0.05]*len(type_counts))
        ax.set_title('Feature Type Distribution', fontsize=16, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        self._save_plot('1_03_feature_types.png')
    
    def plot_1_correlation_heatmap(self):
        """Create correlation heatmap for numerical features"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'dataset_id' in numerical_cols:
            numerical_cols.remove('dataset_id')
        if 'binary_label' in numerical_cols:
            numerical_cols.remove('binary_label')
        
        # Calculate correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        self._save_plot('1_04_correlation_heatmap.png')
    
    def plot_1_feature_distributions(self):
        """Feature distributions (4 features)"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if 'label' not in col.lower() and 'dataset_id' not in col.lower()]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx in range(min(4, len(numeric_cols))):
            col = numeric_cols[idx]
            data = self.df[col].dropna()
            
            if len(data) > 0:
                # Use log scale if data has wide range
                if data.max() > 1000 * data.median():
                    data_plot = np.log10(data + 1)
                    axes[idx].hist(data_plot, bins=50, color='skyblue', 
                                  edgecolor='black', alpha=0.7)
                    axes[idx].set_xlabel(f'Log10({col} + 1)', fontsize=12)
                else:
                    axes[idx].hist(data, bins=50, color='skyblue', 
                                  edgecolor='black', alpha=0.7)
                    axes[idx].set_xlabel(col, fontsize=12)
                
                axes[idx].set_ylabel('Frequency', fontsize=12)
                axes[idx].set_title(f'Distribution: {col[:40]}', fontsize=13, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('1_05_feature_distributions.png')
    
    def plot_1_boxplots_by_class(self):
        """Boxplots for top features grouped by class"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if 'label' not in col.lower() and 'dataset_id' not in col.lower()]
        
        fig, ax = plt.subplots(figsize=(18, 8))
        
        if len(numeric_cols) >= 5:
            # Select top 5 features by variance
            top_5_features = self.df[numeric_cols].var().sort_values(
                ascending=False
            ).head(5).index.tolist()
            
            # Prepare data for boxplot
            plot_data = []
            positions = []
            pos = 0
            
            for feature in top_5_features:
                benign_data = self.df[self.df['binary_label'] == 0][feature].dropna()
                attack_data = self.df[self.df['binary_label'] == 1][feature].dropna()
                
                # Sample if too large
                if len(benign_data) > 10000:
                    benign_data = benign_data.sample(10000, random_state=42)
                if len(attack_data) > 10000:
                    attack_data = attack_data.sample(10000, random_state=42)
                
                plot_data.extend([benign_data, attack_data])
                positions.extend([pos, pos + 0.8])
                pos += 2
            
            bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=False)
            
            # Color boxes
            for i, patch in enumerate(bp['boxes']):
                if i % 2 == 0:
                    patch.set_facecolor('#2ecc71')
                else:
                    patch.set_facecolor('#e74c3c')
            
            ax.set_xticks([i + 0.4 for i in range(0, len(top_5_features)*2, 2)])
            ax.set_xticklabels([f[:20] for f in top_5_features], rotation=45, ha='right')
            ax.set_title('Feature Distributions: BENIGN vs ATTACK (Top 5 by Variance)', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Value (Log Scale)', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            ax.set_yscale('log')
        
        self._save_plot('1_06_boxplots_by_class.png')
    
    # ========================================================================
    # ⚖️ SECTION 2: BINARY CLASSIFICATION PERSPECTIVE
    # ========================================================================
    
    def plot_2_class_balance_bar(self):
        """Class balance bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_counts = self.df['binary_label'].value_counts()
        class_names = {0: 'BENIGN', 1: 'ATTACK'}
        class_counts.index = class_counts.index.map(class_names)
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(class_counts.index, class_counts.values, 
                     color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title('Class Balance: BENIGN vs ATTACK', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        total = class_counts.sum()
        for bar, count in zip(bars, class_counts.values):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, count,
                   f'{count:,}\n({percentage:.2f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        self._save_plot('2_01_class_balance_bar.png')
    
    
    def plot_2_feature_correlation_with_label(self):
        """Feature-label correlation heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['binary_label', 'dataset_id']]
        
        if len(numeric_cols) > 0:
            # Calculate correlation with binary label
            correlations = []
            for col in numeric_cols:
                corr = self.df[col].corr(self.df['binary_label'])
                correlations.append(corr)
            
            corr_df = pd.DataFrame({
                'Feature': numeric_cols,
                'Correlation': correlations
            }).sort_values('Correlation', key=abs, ascending=False).head(20)
            
            colors_corr = ['#e74c3c' if x > 0 else '#3498db' for x in corr_df['Correlation']]
            bars = ax.barh(range(len(corr_df)), corr_df['Correlation'], color=colors_corr)
            ax.set_yticks(range(len(corr_df)))
            ax.set_yticklabels(corr_df['Feature'], fontsize=11)
            ax.set_xlabel('Correlation with Label (ATTACK)', fontsize=14)
            ax.set_title('Top 20 Features by Correlation with Label', 
                        fontsize=16, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            # Store for later use
            self.top_corr_features = corr_df['Feature'].tolist()
        
        self._save_plot('2_03_feature_label_correlation.png')
    
    def plot_2_violin_plots(self):
        """Violin plots for top 4 correlated features"""
        if not hasattr(self, 'top_corr_features'):
            return
        
        top_4_features = self.top_corr_features[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_4_features):
            # Sample data for faster plotting
            sample_df = self.df[[feature, 'binary_label']].dropna()
            if len(sample_df) > 20000:
                sample_df = sample_df.sample(20000, random_state=42)
            
            sample_df['Class'] = sample_df['binary_label'].map({0: 'BENIGN', 1: 'ATTACK'})
            
            sns.violinplot(data=sample_df, x='Class', y=feature, 
                          palette={'BENIGN': '#2ecc71', 'ATTACK': '#e74c3c'},
                          ax=axes[idx])
            axes[idx].set_title(f'{feature[:50]}', fontsize=13, fontweight='bold')
            axes[idx].set_ylabel('Value', fontsize=12)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('2_04_violin_plots_top_features.png')
    
    
    
    
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_complete_analysis(self):
        """Run all analysis sections"""
        print("\n" + "🎯"*40)
        print("COMPREHENSIVE NETWORK INTRUSION DETECTION EDA")
        print("Individual Charts")
        print("🎯"*40 + "\n")
        
        if not self.load_data():
            return False
        
        # Section 1: Data Quality & Structure
        print("\n" + "="*80)
        print("🧩 SECTION 1: DATA QUALITY & STRUCTURE ANALYSIS")
        print("="*80)
        self.plot_1_dataset_sources()
        self.plot_1_missing_values_bar()
        self.plot_1_feature_types()
        self.plot_1_correlation_heatmap()
        self.plot_1_feature_distributions()
        self.plot_1_boxplots_by_class()
        
        # Section 2: Binary Classification
        print("\n" + "="*80)
        print("⚖️ SECTION 2: BINARY CLASSIFICATION ANALYSIS")
        print("="*80)
        self.plot_2_class_balance_bar()
        self.plot_2_feature_correlation_with_label()
        self.plot_2_violin_plots()
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE EDA COMPLETE!")
        print("="*80)
        print(f"\n📊 Total charts generated: {self.chart_count}")
        print(f"📁 All visualizations saved to: {self.output_dir}/")
        print("\nCharts generated:")
        print("  • 1_01: Dataset Sources")
        print("  • 1_02: Missing Values")
        print("  • 1_03: Feature Types")
        print("  • 1_04: Correlation Heatmap")
        print("  • 1_05: Feature Distributions")
        print("  • 1_06: Boxplots by Class")
        print("  • 2_01: Class Balance Bar")
        print("  • 2_03: Feature-Label Correlation")
        print("  • 2_04: Violin Plots")
        
        return True


def main():
    """Main function to run comprehensive EDA"""
    dataset_path = "data/merged_data/merged_datasets.csv"
    output_dir = "EDA/Outputs"
    
    # Create EDA instance and run analysis
    eda = ComprehensiveNetworkEDA(dataset_path, output_dir)
    success = eda.run_complete_analysis()
    
    if success:
        print("\n🎉 All analyses completed successfully!")
        print("📈 Check the EDA/Outputs/ directory for all individual charts.")
    else:
        print("\n❌ Analysis failed. Please check the dataset path and try again.")


if __name__ == "__main__":
    main()
