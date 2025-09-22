#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for Merged Network Intrusion Dataset
Author: Data Scientist - Cyber Security Domain
Date: 2024-09-22

This script performs a complete EDA on the merged_datasets.csv file which combines
CIC-IDS-2017 and UNSW-NB15 datasets for network intrusion detection.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MergedDatasetEDA:
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
    
    def add_to_report(self, text):
        """Add text to the analysis report"""
        self.report_lines.append(text)
        print(text)
    
    def basic_info_analysis(self):
        """Perform basic dataset information analysis"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("BASIC DATASET INFORMATION")
        self.add_to_report("="*80)
        
        # Dataset shape
        self.add_to_report(f"Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Column information
        self.add_to_report(f"\nColumns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = str(self.df[col].dtype)
            self.add_to_report(f"{i:2d}. {col:<20} ({dtype})")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024**2)
        self.add_to_report(f"\nMemory Usage: {memory_usage:.2f} MB")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            self.add_to_report(f"\nMissing Values:")
            for col, missing in missing_values[missing_values > 0].items():
                percentage = (missing / len(self.df)) * 100
                self.add_to_report(f"  {col}: {missing:,} ({percentage:.2f}%)")
        else:
            self.add_to_report("\n✓ No missing values found")
        
        # Data types summary
        dtype_counts = self.df.dtypes.value_counts()
        self.add_to_report(f"\nData Types Summary:")
        for dtype, count in dtype_counts.items():
            self.add_to_report(f"  {dtype}: {count} columns")
    
    def attack_analysis(self):
        """Analyze attack types and distribution"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("ATTACK TYPE ANALYSIS")
        self.add_to_report("="*80)
        
        # Label distribution
        label_counts = self.df['label'].value_counts()
        total_samples = len(self.df)
        
        self.add_to_report(f"Attack Type Distribution:")
        self.add_to_report(f"{'Attack Type':<15} {'Count':<12} {'Percentage':<12}")
        self.add_to_report("-" * 40)
        
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            self.add_to_report(f"{label:<15} {count:<12,} {percentage:<12.2f}%")
        
        # Attack categorization
        attack_categories = {
            'Normal Traffic': ['BENIGN'],
            'Denial of Service': ['DDoS', 'DoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
            'Reconnaissance': ['PortScan', 'Reconnaissance'],
            'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Brute Force'],
            'Web Attacks': ['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'],
            'Malware': ['Bot', 'Backdoor', 'Shellcode', 'Worms'],
            'Other': ['Infiltration', 'Heartbleed', 'Analysis', 'Fuzzers', 'Generic']
        }
        
        self.add_to_report(f"\nAttack Categorization:")
        category_stats = {}
        for category, attacks in attack_categories.items():
            category_count = sum(label_counts.get(attack, 0) for attack in attacks)
            if category_count > 0:
                category_percentage = (category_count / total_samples) * 100
                category_stats[category] = category_count
                self.add_to_report(f"{category:<20}: {category_count:>10,} ({category_percentage:>6.2f}%)")
        
        # Dataset source analysis
        if 'dataset_id' in self.df.columns:
            self.add_to_report(f"\nDataset Source Distribution:")
            dataset_counts = self.df['dataset_id'].value_counts()
            for dataset_id, count in dataset_counts.items():
                percentage = (count / total_samples) * 100
                # Handle mixed data types in dataset_id
                if str(dataset_id) == '1' or dataset_id == 1:
                    dataset_name = "CIC-IDS-2017"
                elif str(dataset_id) == '0' or dataset_id == 0:
                    dataset_name = "UNSW-NB15"
                else:
                    dataset_name = f"Dataset {dataset_id}"
                self.add_to_report(f"  {dataset_name}: {count:,} ({percentage:.2f}%)")
    
    def network_flow_analysis(self):
        """Analyze network flow characteristics"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("NETWORK FLOW CHARACTERISTICS")
        self.add_to_report("="*80)
        
        # Protocol analysis
        if 'protocol' in self.df.columns:
            protocol_counts = self.df['protocol'].value_counts()
            self.add_to_report(f"Protocol Distribution:")
            for protocol, count in protocol_counts.items():
                percentage = (count / len(self.df)) * 100
                self.add_to_report(f"  {protocol.upper():<8}: {count:>10,} ({percentage:>6.2f}%)")
        
        # Connection state analysis
        if 'state' in self.df.columns:
            state_counts = self.df['state'].value_counts()
            self.add_to_report(f"\nConnection State Distribution:")
            for state, count in state_counts.items():
                percentage = (count / len(self.df)) * 100
                self.add_to_report(f"  {state:<8}: {count:>10,} ({percentage:>6.2f}%)")
        
        # Duration analysis
        if 'duration' in self.df.columns:
            duration_stats = self.df['duration'].describe()
            self.add_to_report(f"\nConnection Duration Statistics (seconds):")
            self.add_to_report(f"  Mean:     {duration_stats['mean']:.6f}")
            self.add_to_report(f"  Median:   {duration_stats['50%']:.6f}")
            self.add_to_report(f"  Std Dev:  {duration_stats['std']:.6f}")
            self.add_to_report(f"  Min:      {duration_stats['min']:.6f}")
            self.add_to_report(f"  Max:      {duration_stats['max']:.6f}")
            
            # Duration by attack type
            duration_by_attack = self.df.groupby('label')['duration'].agg(['mean', 'median', 'std']).round(6)
            self.add_to_report(f"\nDuration by Attack Type:")
            self.add_to_report(f"{'Attack Type':<15} {'Mean':<12} {'Median':<12} {'Std Dev':<12}")
            self.add_to_report("-" * 55)
            for attack, stats in duration_by_attack.iterrows():
                self.add_to_report(f"{attack:<15} {stats['mean']:<12.6f} {stats['median']:<12.6f} {stats['std']:<12.6f}")
    
    def port_analysis(self):
        """Analyze destination port patterns"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("PORT ANALYSIS")
        self.add_to_report("="*80)
        
        if 'destination_port' in self.df.columns:
            # Top destination ports
            port_counts = self.df['destination_port'].value_counts().head(20)
            self.add_to_report(f"Top 20 Destination Ports:")
            self.add_to_report(f"{'Port':<8} {'Count':<12} {'Percentage':<12} {'Common Service'}")
            self.add_to_report("-" * 60)
            
            # Common port mappings
            port_services = {
                21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP', 53: 'DNS',
                80: 'HTTP', 110: 'POP3', 139: 'NetBIOS', 143: 'IMAP', 443: 'HTTPS',
                993: 'IMAPS', 995: 'POP3S', 1433: 'SQL Server', 3389: 'RDP',
                5432: 'PostgreSQL', 3306: 'MySQL', 8080: 'HTTP-Alt', 8443: 'HTTPS-Alt'
            }
            
            for port, count in port_counts.items():
                percentage = (count / len(self.df)) * 100
                service = port_services.get(port, 'Unknown/Custom')
                self.add_to_report(f"{port:<8} {count:<12,} {percentage:<12.2f}% {service}")
            
            # Port analysis by attack type
            self.add_to_report(f"\nPort Usage by Attack Type (Top 5 ports per attack):")
            for attack_type in self.df['label'].unique():
                attack_data = self.df[self.df['label'] == attack_type]
                top_ports = attack_data['destination_port'].value_counts().head(5)
                if len(top_ports) > 0:
                    self.add_to_report(f"\n{attack_type}:")
                    for port, count in top_ports.items():
                        percentage = (count / len(attack_data)) * 100
                        service = port_services.get(port, 'Unknown')
                        self.add_to_report(f"  Port {port} ({service}): {count:,} ({percentage:.1f}%)")
    
    def bytes_packets_analysis(self):
        """Analyze bytes and packets statistics"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("BYTES AND PACKETS ANALYSIS")
        self.add_to_report("="*80)
        
        # Bytes analysis
        byte_columns = ['source_bytes', 'dest_bytes']
        packet_columns = ['source_pkts', 'dest_pkts']
        
        for col in byte_columns + packet_columns:
            if col in self.df.columns:
                stats = self.df[col].describe()
                unit = "bytes" if "bytes" in col else "packets"
                self.add_to_report(f"\n{col.replace('_', ' ').title()} Statistics ({unit}):")
                self.add_to_report(f"  Mean:     {stats['mean']:.2f}")
                self.add_to_report(f"  Median:   {stats['50%']:.2f}")
                self.add_to_report(f"  Std Dev:  {stats['std']:.2f}")
                self.add_to_report(f"  Min:      {stats['min']:.2f}")
                self.add_to_report(f"  Max:      {stats['max']:.2f}")
        
        # Traffic volume by attack type
        if all(col in self.df.columns for col in ['source_bytes', 'dest_bytes', 'label']):
            self.df['total_bytes'] = self.df['source_bytes'] + self.df['dest_bytes']
            traffic_by_attack = self.df.groupby('label')['total_bytes'].agg(['mean', 'median', 'sum']).round(2)
            
            self.add_to_report(f"\nTraffic Volume by Attack Type:")
            self.add_to_report(f"{'Attack Type':<15} {'Mean Bytes':<15} {'Median Bytes':<15} {'Total Bytes':<15}")
            self.add_to_report("-" * 65)
            for attack, stats in traffic_by_attack.iterrows():
                self.add_to_report(f"{attack:<15} {stats['mean']:<15,.0f} {stats['median']:<15,.0f} {stats['sum']:<15,.0f}")
    
    def tcp_window_analysis(self):
        """Analyze TCP window sizes"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("TCP WINDOW ANALYSIS")
        self.add_to_report("="*80)
        
        tcp_columns = ['tcp_win_fwd', 'tcp_win_bwd']
        
        for col in tcp_columns:
            if col in self.df.columns:
                # Filter out -1 values which indicate non-TCP traffic
                tcp_data = self.df[self.df[col] != -1][col]
                if len(tcp_data) > 0:
                    stats = tcp_data.describe()
                    direction = "Forward" if "fwd" in col else "Backward"
                    self.add_to_report(f"\nTCP Window Size - {direction} Direction:")
                    self.add_to_report(f"  Count:    {len(tcp_data):,}")
                    self.add_to_report(f"  Mean:     {stats['mean']:.2f}")
                    self.add_to_report(f"  Median:   {stats['50%']:.2f}")
                    self.add_to_report(f"  Std Dev:  {stats['std']:.2f}")
                    self.add_to_report(f"  Min:      {stats['min']:.2f}")
                    self.add_to_report(f"  Max:      {stats['max']:.2f}")
    
    def segment_size_analysis(self):
        """Analyze mean segment sizes"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("SEGMENT SIZE ANALYSIS")
        self.add_to_report("="*80)
        
        segment_columns = ['mean_seg_size_fwd', 'mean_seg_size_bwd']
        
        for col in segment_columns:
            if col in self.df.columns:
                stats = self.df[col].describe()
                direction = "Forward" if "fwd" in col else "Backward"
                self.add_to_report(f"\nMean Segment Size - {direction} Direction:")
                self.add_to_report(f"  Mean:     {stats['mean']:.2f}")
                self.add_to_report(f"  Median:   {stats['50%']:.2f}")
                self.add_to_report(f"  Std Dev:  {stats['std']:.2f}")
                self.add_to_report(f"  Min:      {stats['min']:.2f}")
                self.add_to_report(f"  Max:      {stats['max']:.2f}")
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("CORRELATION ANALYSIS")
        self.add_to_report("="*80)
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'dataset_id' in numerical_cols:
            numerical_cols.remove('dataset_id')
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            
            # Find high correlations (> 0.7 or < -0.7)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            if high_correlations:
                self.add_to_report(f"High Correlations (|r| > 0.7):")
                self.add_to_report(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation':<12}")
                self.add_to_report("-" * 55)
                for feat1, feat2, corr in sorted(high_correlations, key=lambda x: abs(x[2]), reverse=True):
                    self.add_to_report(f"{feat1:<20} {feat2:<20} {corr:<12.3f}")
            else:
                self.add_to_report("No high correlations (|r| > 0.7) found between features.")
    
    def data_quality_assessment(self):
        """Assess data quality issues"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("DATA QUALITY ASSESSMENT")
        self.add_to_report("="*80)
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            percentage = (duplicate_count / len(self.df)) * 100
            self.add_to_report(f"⚠ Duplicate rows: {duplicate_count:,} ({percentage:.2f}%)")
        else:
            self.add_to_report("✓ No duplicate rows found")
        
        # Check for infinite values
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numerical_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            self.add_to_report(f"\n⚠ Infinite values found:")
            for col, count in inf_counts.items():
                percentage = (count / len(self.df)) * 100
                self.add_to_report(f"  {col}: {count:,} ({percentage:.2f}%)")
        else:
            self.add_to_report("\n✓ No infinite values found")
        
        # Check for negative values in columns that shouldn't have them
        non_negative_cols = ['source_bytes', 'dest_bytes', 'source_pkts', 'dest_pkts', 'duration']
        negative_issues = {}
        for col in non_negative_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    negative_issues[col] = negative_count
        
        if negative_issues:
            self.add_to_report(f"\n⚠ Negative values in non-negative columns:")
            for col, count in negative_issues.items():
                percentage = (count / len(self.df)) * 100
                self.add_to_report(f"  {col}: {count:,} ({percentage:.2f}%)")
        else:
            self.add_to_report(f"\n✓ No negative values in non-negative columns")
    
    def cybersecurity_insights(self):
        """Generate cybersecurity-specific insights"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("CYBERSECURITY INSIGHTS")
        self.add_to_report("="*80)
        
        # Attack patterns by protocol
        if all(col in self.df.columns for col in ['protocol', 'label']):
            attack_protocol = pd.crosstab(self.df['label'], self.df['protocol'], normalize='index') * 100
            self.add_to_report(f"Attack Distribution by Protocol (%):")
            self.add_to_report(attack_protocol.round(1).to_string())
        
        # Suspicious port activity
        if 'destination_port' in self.df.columns:
            # High-risk ports (commonly targeted)
            high_risk_ports = [21, 22, 23, 135, 139, 445, 1433, 3389, 5432]
            high_risk_traffic = self.df[self.df['destination_port'].isin(high_risk_ports)]
            
            if len(high_risk_traffic) > 0:
                risk_percentage = (len(high_risk_traffic) / len(self.df)) * 100
                self.add_to_report(f"\nHigh-Risk Port Activity:")
                self.add_to_report(f"  Traffic to high-risk ports: {len(high_risk_traffic):,} ({risk_percentage:.2f}%)")
                
                # Attack distribution on high-risk ports
                risk_attacks = high_risk_traffic['label'].value_counts()
                self.add_to_report(f"  Attack distribution on high-risk ports:")
                for attack, count in risk_attacks.items():
                    percentage = (count / len(high_risk_traffic)) * 100
                    self.add_to_report(f"    {attack}: {count:,} ({percentage:.1f}%)")
        
        # Connection state anomalies
        if all(col in self.df.columns for col in ['state', 'label']):
            # Analyze connection states for different attack types
            state_attack = pd.crosstab(self.df['state'], self.df['label'], normalize='columns') * 100
            self.add_to_report(f"\nConnection State Distribution by Attack Type (%):")
            self.add_to_report(state_attack.round(1).to_string())
    
    def generate_summary(self):
        """Generate executive summary"""
        self.add_to_report("\n" + "="*80)
        self.add_to_report("EXECUTIVE SUMMARY")
        self.add_to_report("="*80)
        
        total_samples = len(self.df)
        attack_samples = len(self.df[self.df['label'] != 'BENIGN'])
        attack_percentage = (attack_samples / total_samples) * 100
        
        self.add_to_report(f"Dataset Overview:")
        self.add_to_report(f"• Total network flows analyzed: {total_samples:,}")
        self.add_to_report(f"• Malicious flows detected: {attack_samples:,} ({attack_percentage:.1f}%)")
        self.add_to_report(f"• Benign flows: {total_samples - attack_samples:,} ({100 - attack_percentage:.1f}%)")
        
        # Attack type summary
        attack_types = self.df[self.df['label'] != 'BENIGN']['label'].nunique()
        self.add_to_report(f"• Unique attack types: {attack_types}")
        
        # Protocol summary
        if 'protocol' in self.df.columns:
            protocols = self.df['protocol'].nunique()
            self.add_to_report(f"• Network protocols observed: {protocols}")
        
        # Port summary
        if 'destination_port' in self.df.columns:
            unique_ports = self.df['destination_port'].nunique()
            self.add_to_report(f"• Unique destination ports: {unique_ports:,}")
        
        self.add_to_report(f"\nKey Findings:")
        self.add_to_report(f"• This merged dataset combines network intrusion data from multiple sources")
        self.add_to_report(f"• The dataset is suitable for training machine learning models for intrusion detection")
        self.add_to_report(f"• Features include network flow characteristics, timing, and traffic volume metrics")
        self.add_to_report(f"• The data shows diverse attack patterns across different protocols and ports")
    
    def save_report(self):
        """Save the analysis report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eda_outputs/Merged_Dataset_EDA_Analysis_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Merged Network Intrusion Dataset - EDA Report\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.file_path}\n")
                f.write("\n".join(self.report_lines))
            
            print(f"\n✓ Analysis report saved to: {filename}")
            return filename
        except Exception as e:
            print(f"✗ Error saving report: {e}")
            return None
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis"""
        print("Starting Comprehensive EDA for Merged Network Intrusion Dataset")
        print("=" * 70)
        
        if not self.load_data():
            return False
        
        # Run all analysis components
        self.basic_info_analysis()
        self.attack_analysis()
        self.network_flow_analysis()
        self.port_analysis()
        self.bytes_packets_analysis()
        self.tcp_window_analysis()
        self.segment_size_analysis()
        self.correlation_analysis()
        self.data_quality_assessment()
        self.cybersecurity_insights()
        self.generate_summary()
        
        # Save report
        report_file = self.save_report()
        
        print("\n" + "="*70)
        print("EDA Analysis Complete!")
        print("="*70)
        
        return True

def main():
    """Main function to run the EDA"""
    # Dataset path
    dataset_path = "processed_data/merged_datasets.csv"
    
    # Create EDA instance and run analysis
    eda = MergedDatasetEDA(dataset_path)
    success = eda.run_complete_analysis()
    
    if success:
        print("\n🎉 Merged Dataset EDA completed successfully!")
        print("📊 Check the eda_outputs/ directory for the detailed report.")
    else:
        print("\n❌ EDA analysis failed. Please check the dataset path and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
