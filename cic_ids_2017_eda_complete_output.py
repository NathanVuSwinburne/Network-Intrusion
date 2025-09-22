#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for CIC-IDS-2017 Dataset
This script performs detailed analysis and saves ALL output to a comprehensive file.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import sys
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class OutputCapture:
    """Capture both console output and save to file"""
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        
    def save_to_file(self):
        output_dir = Path("eda_outputs")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / self.filename, 'w', encoding='utf-8') as f:
            f.write(self.log.getvalue())

class CICIDSAnalyzer:
    """
    Comprehensive analyzer for CIC-IDS-2017 dataset with complete output capture
    """
    
    def __init__(self, dataset_path="network-intrusion-dataset/CIC_IDS_2017"):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.file_info = {}
        self.attack_categories = {
            'BENIGN': 'Normal Traffic',
            'DDoS': 'Denial of Service',
            'PortScan': 'Port Scanning',
            'Bot': 'Botnet Activity',
            'Infiltration': 'Infiltration Attack',
            'Web Attack – Brute Force': 'Web Attacks',
            'Web Attack – XSS': 'Web Attacks',
            'Web Attack – Sql Injection': 'Web Attacks',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'DoS Hulk': 'Denial of Service',
            'DoS GoldenEye': 'Denial of Service',
            'DoS slowloris': 'Denial of Service',
            'DoS Slowhttptest': 'Denial of Service',
            'Heartbleed': 'Vulnerability Exploit'
        }
        
    def load_dataset(self):
        """Load and combine all CSV files from the CIC-IDS-2017 dataset"""
        print("🔄 Loading CIC-IDS-2017 dataset...")
        
        csv_files = list(self.dataset_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")
        
        dataframes = []
        
        for file_path in csv_files:
            print(f"   📁 Loading {file_path.name}...")
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.name
                
                # Store file information
                self.file_info[file_path.name] = {
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                
                dataframes.append(df)
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")
        
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            print(f"✅ Successfully loaded {len(dataframes)} files")
            print(f"📊 Total dataset shape: {self.data.shape}")
        else:
            raise ValueError("No data could be loaded from the dataset")
    
    def basic_info_analysis(self):
        """Perform basic dataset information analysis"""
        print("\n" + "="*60)
        print("📋 BASIC DATASET INFORMATION")
        print("="*60)
        
        # Dataset overview
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Number of Files: {len(self.file_info)}")
        
        # File breakdown
        print("\n📁 File Information:")
        for filename, info in self.file_info.items():
            print(f"   {filename}: {info['rows']:,} rows, {info['size_mb']:.1f} MB")
        
        # Column information
        print(f"\n📊 Columns ({len(self.data.columns)}):")
        print(f"   Numeric columns: {len(self.data.select_dtypes(include=[np.number]).columns)}")
        print(f"   Object columns: {len(self.data.select_dtypes(include=['object']).columns)}")
        
        # Show first few column names
        print(f"\n📝 Sample Column Names:")
        for i, col in enumerate(self.data.columns[:10]):
            print(f"   {i+1:2d}. {col}")
        if len(self.data.columns) > 10:
            print(f"   ... and {len(self.data.columns) - 10} more columns")
        
        # Missing values analysis
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        if missing_data.sum() > 0:
            print(f"\n⚠️  Missing Values:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_data[missing_data > 0],
                'Percentage': missing_percent[missing_data > 0]
            }).sort_values('Missing Count', ascending=False)
            print(missing_df.head(10).to_string())
        else:
            print("\n✅ No missing values found")
        
        # Data types
        print(f"\n🔢 Data Types:")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
    
    def attack_distribution_analysis(self):
        """Analyze the distribution of attack types"""
        print("\n" + "="*60)
        print("🎯 ATTACK TYPE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Identify label column (common names)
        label_col = None
        for col in [' Label', 'Label', 'label', ' label']:
            if col in self.data.columns:
                label_col = col
                break
        
        if label_col is None:
            print("❌ Label column not found")
            print("Available columns containing 'label':")
            label_like_cols = [col for col in self.data.columns if 'label' in col.lower()]
            for col in label_like_cols:
                print(f"   - {col}")
            return None, None
        
        print(f"✅ Using label column: '{label_col}'")
        
        # Clean label data
        self.data[label_col] = self.data[label_col].str.strip()
        
        # Attack type distribution
        attack_counts = self.data[label_col].value_counts()
        attack_percentages = (attack_counts / len(self.data)) * 100
        
        print(f"\n📊 Attack Type Distribution (All {len(attack_counts)} types):")
        print("-" * 80)
        for attack, count in attack_counts.items():
            percentage = attack_percentages[attack]
            category = self.attack_categories.get(attack, 'Other')
            print(f"   {attack:<35} | {count:>10,} ({percentage:>5.2f}%) | {category}")
        
        # Category-wise analysis
        print(f"\n🏷️  Attack Categories Summary:")
        print("-" * 60)
        category_counts = {}
        for attack, count in attack_counts.items():
            category = self.attack_categories.get(attack, 'Other')
            category_counts[category] = category_counts.get(category, 0) + count
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.data)) * 100
            print(f"   {category:<30} | {count:>10,} ({percentage:>5.2f}%)")
        
        return attack_counts, label_col
    
    def network_flow_analysis(self):
        """Analyze network flow characteristics"""
        print("\n" + "="*60)
        print("🌐 NETWORK FLOW CHARACTERISTICS")
        print("="*60)
        
        # Find relevant columns
        flow_columns = []
        for col in self.data.columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in ['flow', 'duration', 'packet', 'byte', 'length']):
                flow_columns.append(col)
        
        if not flow_columns:
            print("❌ No flow-related columns found")
            return
        
        print(f"📊 Flow-related columns found: {len(flow_columns)}")
        print("   Flow columns:")
        for i, col in enumerate(flow_columns[:15], 1):
            print(f"   {i:2d}. {col}")
        if len(flow_columns) > 15:
            print(f"   ... and {len(flow_columns) - 15} more flow columns")
        
        # Analyze numeric flow columns
        numeric_flow_cols = []
        for col in flow_columns:
            if self.data[col].dtype in ['int64', 'float64']:
                numeric_flow_cols.append(col)
        
        if numeric_flow_cols:
            print(f"\n🔢 Numeric Flow Statistics (Top 10 columns):")
            print("-" * 100)
            # Show statistics for first 10 numeric flow columns
            top_flow_cols = numeric_flow_cols[:10]
            flow_stats = self.data[top_flow_cols].describe()
            print(flow_stats.to_string())
            
            # Zero values analysis
            print(f"\n🔍 Zero Values Analysis:")
            print("-" * 70)
            for col in numeric_flow_cols[:15]:  # Limit to first 15 columns
                zero_count = (self.data[col] == 0).sum()
                zero_percent = (zero_count / len(self.data)) * 100
                print(f"   {col:<45}: {zero_count:>8,} ({zero_percent:>5.1f}%)")
    
    def port_analysis(self):
        """Analyze destination port patterns"""
        print("\n" + "="*60)
        print("🔌 PORT ANALYSIS")
        print("="*60)
        
        # Find port columns
        port_cols = []
        for col in self.data.columns:
            col_lower = col.lower().strip()
            if 'port' in col_lower:
                port_cols.append(col)
        
        if not port_cols:
            print("❌ No port columns found")
            return
        
        print(f"📊 Port columns found: {port_cols}")
        
        # Analyze destination ports
        dest_port_col = None
        for col in port_cols:
            if 'destination' in col.lower() or 'dest' in col.lower():
                dest_port_col = col
                break
        
        if dest_port_col is None and port_cols:
            dest_port_col = port_cols[0]  # Use first port column
        
        if dest_port_col:
            print(f"🎯 Analyzing column: {dest_port_col}")
            
            # Top destination ports
            top_ports = self.data[dest_port_col].value_counts().head(25)
            
            # Common port mappings
            port_services = {
                80: 'HTTP', 443: 'HTTPS', 22: 'SSH', 21: 'FTP', 25: 'SMTP',
                53: 'DNS', 110: 'POP3', 143: 'IMAP', 993: 'IMAPS', 995: 'POP3S',
                23: 'Telnet', 3389: 'RDP', 135: 'RPC', 139: 'NetBIOS', 445: 'SMB',
                3306: 'MySQL', 5432: 'PostgreSQL', 1433: 'MSSQL', 6379: 'Redis'
            }
            
            print(f"\n🔝 Top 25 Destination Ports:")
            print("-" * 70)
            for port, count in top_ports.items():
                service = port_services.get(port, 'Unknown')
                percentage = (count / len(self.data)) * 100
                print(f"   Port {port:<6} | {service:<12} | {count:>10,} ({percentage:>5.2f}%)")
    
    def feature_correlation_analysis(self):
        """Analyze feature correlations"""
        print("\n" + "="*60)
        print("🔗 FEATURE CORRELATION ANALYSIS")
        print("="*60)
        
        # Get numeric columns only
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("❌ Not enough numeric columns for correlation analysis")
            return
        
        print(f"📊 Analyzing correlations for {len(numeric_cols)} numeric features")
        
        # Calculate correlation matrix for a subset (performance reasons)
        sample_cols = numeric_cols[:20] if len(numeric_cols) > 20 else numeric_cols
        print(f"🔍 Using sample of {len(sample_cols)} columns for correlation analysis")
        
        corr_matrix = self.data[sample_cols].corr()
        
        # Find highly correlated pairs (> 0.8 or < -0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"\n🔍 Highly Correlated Feature Pairs (|correlation| > 0.8):")
            print("-" * 90)
            for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {feat1:<30} ↔ {feat2:<30} | {corr_val:>6.3f}")
        else:
            print("\n✅ No highly correlated feature pairs found (threshold: 0.8)")
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis pipeline"""
        print("🚀 CIC-IDS-2017 Dataset - Complete EDA Analysis")
        print("=" * 60)
        print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Load dataset
            self.load_dataset()
            
            # Basic information analysis
            self.basic_info_analysis()
            
            # Attack distribution analysis
            attack_counts, label_col = self.attack_distribution_analysis()
            
            # Network flow analysis
            self.network_flow_analysis()
            
            # Port analysis
            self.port_analysis()
            
            # Feature correlation analysis
            self.feature_correlation_analysis()
            
            print("\n" + "="*60)
            print("✅ COMPLETE ANALYSIS FINISHED!")
            print("="*60)
            print(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("📁 All output has been saved to the eda_outputs directory")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

def main():
    """Main execution function with output capture"""
    # Set up output capture
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"CIC_IDS_2017_Complete_Analysis_{timestamp}.txt"
    
    # Capture all output
    output_capture = OutputCapture(output_filename)
    sys.stdout = output_capture
    
    try:
        print("CIC-IDS-2017 Dataset - Complete EDA Analysis with Full Output Capture")
        print("=" * 70)
        
        # Initialize analyzer
        analyzer = CICIDSAnalyzer()
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    finally:
        # Restore stdout and save output
        sys.stdout = output_capture.terminal
        output_capture.save_to_file()
        print(f"\n✅ Complete analysis output saved to: eda_outputs/{output_filename}")

if __name__ == "__main__":
    main()
