#!/usr/bin/env python3
"""
Enhanced EDA with Visualizations for Merged Network Intrusion Dataset
Author: Data Scientist - Cyber Security Domain
Date: 2024-09-22

This script performs comprehensive EDA with visualizations on merged_datasets.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MergedDatasetEDAWithPlots:
    def __init__(self, file_path):
        """Initialize the EDA class with dataset path"""
        self.file_path = file_path
        self.df = None
        self.report_lines = []
        
    def load_data(self):
        """Load the merged dataset"""
        print("Loading merged dataset...")
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset loaded successfully: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def create_attack_distribution_plot(self):
        """Create attack type distribution visualization"""
        plt.figure(figsize=(15, 10))
        
        # Get top 15 attack types for better visualization
        attack_counts = self.df['label'].value_counts().head(15)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Bar plot of attack distribution
        attack_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Top 15 Attack Types Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Attack Type', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(attack_counts.values):
            ax1.text(i, v + max(attack_counts.values) * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Pie chart for attack categories
        attack_categories = {
            'Normal Traffic': ['BENIGN'],
            'Denial of Service': ['DDoS', 'DoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
            'Reconnaissance': ['PortScan', 'Reconnaissance'],
            'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Brute Force'],
            'Web Attacks': ['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'],
            'Malware': ['Bot', 'Backdoor', 'Shellcode', 'Worms'],
            'Other': ['Infiltration', 'Heartbleed', 'Analysis', 'Fuzzers', 'Generic', 'Exploits']
        }
        
        category_counts = []
        category_labels = []
        all_attacks = self.df['label'].value_counts()
        
        for category, attacks in attack_categories.items():
            count = sum(all_attacks.get(attack, 0) for attack in attacks)
            if count > 0:
                category_counts.append(count)
                category_labels.append(f'{category}\n({count:,})')
        
        ax2.pie(category_counts, labels=category_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Attack Categories Distribution', fontsize=14, fontweight='bold')
        
        # 3. Protocol distribution
        protocol_counts = self.df['protocol'].value_counts().head(10)
        protocol_counts.plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='black')
        ax3.set_title('Top 10 Protocol Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Protocol', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Connection state distribution
        state_counts = self.df['state'].value_counts()
        state_counts.plot(kind='bar', ax=ax4, color='lightgreen', edgecolor='black')
        ax4.set_title('Connection State Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Connection State', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_outputs/attack_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Attack distribution plots saved to: eda_outputs/attack_distribution_analysis.png")
    
    def create_port_analysis_plots(self):
        """Create port analysis visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Top destination ports
        top_ports = self.df['destination_port'].value_counts().head(20)
        top_ports.plot(kind='bar', ax=ax1, color='orange', edgecolor='black')
        ax1.set_title('Top 20 Destination Ports', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Port Number', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Port usage by attack type (heatmap)
        # Get top 10 ports and top 10 attacks for heatmap
        top_10_ports = self.df['destination_port'].value_counts().head(10).index
        top_10_attacks = self.df['label'].value_counts().head(10).index
        
        port_attack_matrix = pd.crosstab(
            self.df[self.df['destination_port'].isin(top_10_ports)]['destination_port'],
            self.df[self.df['label'].isin(top_10_attacks)]['label']
        )
        
        sns.heatmap(port_attack_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Port vs Attack Type Heatmap (Top 10 each)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Attack Type', fontsize=12)
        ax2.set_ylabel('Destination Port', fontsize=12)
        
        # 3. High-risk ports analysis
        high_risk_ports = [21, 22, 23, 53, 80, 135, 139, 443, 445, 1433, 3389, 5432]
        risk_port_data = self.df[self.df['destination_port'].isin(high_risk_ports)]
        
        if len(risk_port_data) > 0:
            risk_counts = risk_port_data['destination_port'].value_counts()
            risk_counts.plot(kind='bar', ax=ax3, color='red', alpha=0.7, edgecolor='black')
            ax3.set_title('High-Risk Ports Traffic', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Port Number', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Port distribution by protocol
        port_protocol = self.df.groupby('protocol')['destination_port'].nunique().sort_values(ascending=False)
        port_protocol.plot(kind='bar', ax=ax4, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_title('Unique Ports by Protocol', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Protocol', fontsize=12)
        ax4.set_ylabel('Number of Unique Ports', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_outputs/port_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Port analysis plots saved to: eda_outputs/port_analysis.png")
    
    def create_traffic_analysis_plots(self):
        """Create traffic volume and timing analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Traffic volume distribution (log scale)
        self.df['total_bytes'] = self.df['source_bytes'] + self.df['dest_bytes']
        traffic_data = self.df[self.df['total_bytes'] > 0]['total_bytes']
        
        ax1.hist(np.log10(traffic_data + 1), bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Traffic Volume Distribution (Log Scale)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Log10(Total Bytes + 1)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # 2. Duration distribution (log scale)
        duration_data = self.df[self.df['duration'] > 0]['duration']
        if len(duration_data) > 0:
            ax2.hist(np.log10(duration_data), bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            ax2.set_title('Connection Duration Distribution (Log Scale)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Log10(Duration in seconds)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
        
        # 3. Traffic by attack type (box plot)
        top_attacks = self.df['label'].value_counts().head(8).index
        attack_traffic_data = self.df[self.df['label'].isin(top_attacks)]
        
        sns.boxplot(data=attack_traffic_data, x='label', y='total_bytes', ax=ax3)
        ax3.set_title('Traffic Volume by Attack Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Attack Type', fontsize=12)
        ax3.set_ylabel('Total Bytes', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        # 4. Packet count analysis
        self.df['total_packets'] = self.df['source_pkts'] + self.df['dest_pkts']
        packet_data = self.df[self.df['total_packets'] > 0]['total_packets']
        
        ax4.hist(np.log10(packet_data), bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_title('Packet Count Distribution (Log Scale)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Log10(Total Packets)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('eda_outputs/traffic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Traffic analysis plots saved to: eda_outputs/traffic_analysis.png")
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical features"""
        plt.figure(figsize=(12, 10))
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'dataset_id' in numerical_cols:
            numerical_cols.remove('dataset_id')
        
        # Calculate correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('eda_outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Correlation heatmap saved to: eda_outputs/correlation_heatmap.png")
    
    def create_cybersecurity_insights_plots(self):
        """Create cybersecurity-specific visualization insights"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Attack timeline simulation (using duration as proxy)
        attack_data = self.df[self.df['label'] != 'BENIGN'].copy()
        if len(attack_data) > 0:
            attack_duration = attack_data.groupby('label')['duration'].mean().sort_values(ascending=False).head(10)
            attack_duration.plot(kind='bar', ax=ax1, color='red', alpha=0.7, edgecolor='black')
            ax1.set_title('Average Attack Duration by Type', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Attack Type', fontsize=12)
            ax1.set_ylabel('Average Duration (seconds)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Protocol security analysis
        protocol_attack = pd.crosstab(self.df['protocol'], self.df['label'] != 'BENIGN')
        protocol_attack_pct = protocol_attack.div(protocol_attack.sum(axis=1), axis=0) * 100
        
        top_protocols = self.df['protocol'].value_counts().head(10).index
        protocol_risk = protocol_attack_pct.loc[top_protocols, True].sort_values(ascending=False)
        
        protocol_risk.plot(kind='bar', ax=ax2, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_title('Attack Percentage by Protocol', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Protocol', fontsize=12)
        ax2.set_ylabel('Attack Percentage (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. TCP Window analysis for attacks
        tcp_data = self.df[self.df['tcp_win_fwd'] != -1].copy()
        if len(tcp_data) > 0:
            tcp_attack_comparison = tcp_data.groupby(tcp_data['label'] != 'BENIGN')['tcp_win_fwd'].mean()
            tcp_attack_comparison.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax3.set_title('Average TCP Window Size: Benign vs Attacks', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Traffic Type', fontsize=12)
            ax3.set_ylabel('Average TCP Window Size', fontsize=12)
            ax3.set_xticklabels(['Benign', 'Attack'], rotation=0)
        
        # 4. Segment size analysis
        segment_data = self.df[['mean_seg_size_fwd', 'mean_seg_size_bwd', 'label']].copy()
        segment_data['is_attack'] = segment_data['label'] != 'BENIGN'
        
        segment_comparison = segment_data.groupby('is_attack')[['mean_seg_size_fwd', 'mean_seg_size_bwd']].mean()
        segment_comparison.plot(kind='bar', ax=ax4, alpha=0.7, edgecolor='black')
        ax4.set_title('Average Segment Sizes: Benign vs Attacks', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Traffic Type', fontsize=12)
        ax4.set_ylabel('Average Segment Size', fontsize=12)
        ax4.set_xticklabels(['Benign', 'Attack'], rotation=0)
        ax4.legend(['Forward', 'Backward'])
        
        plt.tight_layout()
        plt.savefig('eda_outputs/cybersecurity_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Cybersecurity insights plots saved to: eda_outputs/cybersecurity_insights.png")
    
    def run_complete_analysis_with_plots(self):
        """Run the complete EDA analysis with visualizations"""
        print("Starting Enhanced EDA with Visualizations for Merged Network Intrusion Dataset")
        print("=" * 80)
        
        if not self.load_data():
            return False
        
        print("\n🎨 Creating visualizations...")
        
        # Create all visualization plots
        self.create_attack_distribution_plot()
        self.create_port_analysis_plots()
        self.create_traffic_analysis_plots()
        self.create_correlation_heatmap()
        self.create_cybersecurity_insights_plots()
        
        print("\n" + "="*80)
        print("Enhanced EDA with Visualizations Complete!")
        print("="*80)
        print("📊 All plots have been saved to the eda_outputs/ directory:")
        print("  • attack_distribution_analysis.png")
        print("  • port_analysis.png") 
        print("  • traffic_analysis.png")
        print("  • correlation_heatmap.png")
        print("  • cybersecurity_insights.png")
        
        return True

def main():
    """Main function to run the enhanced EDA with plots"""
    # Dataset path
    dataset_path = "processed_data/merged_datasets.csv"
    
    # Create EDA instance and run analysis
    eda = MergedDatasetEDAWithPlots(dataset_path)
    success = eda.run_complete_analysis_with_plots()
    
    if success:
        print("\n🎉 Enhanced EDA with visualizations completed successfully!")
        print("📈 Check the eda_outputs/ directory for all generated plots and insights.")
    else:
        print("\n❌ EDA analysis failed. Please check the dataset path and try again.")

if __name__ == "__main__":
    main()
