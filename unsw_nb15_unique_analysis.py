#!/usr/bin/env python3
"""
UNSW-NB15 Unique Values and Attack Distribution Analysis
This script analyzes all unique values in each column, their percentages,
and provides comprehensive attack distribution analysis.
Results are saved to eda_outputs directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from io import StringIO

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

class UNSWUniqueAnalyzer:
    """
    Comprehensive unique values analyzer for UNSW-NB15 dataset
    """
    
    def __init__(self, dataset_path="network-intrusion-dataset/UNSW_NB15"):
        self.dataset_path = Path(dataset_path)
        self.data = None
        
        # UNSW-NB15 column definitions
        self.column_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
            'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
            'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit',
            'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
            'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        
        # Attack category descriptions
        self.attack_descriptions = {
            'Normal': 'Normal/Benign Traffic',
            'Generic': 'Generic Attack Patterns',
            'Exploits': 'Exploitation Attacks',
            'Fuzzers': 'Fuzzing Attacks',
            'DoS': 'Denial of Service',
            'Reconnaissance': 'Reconnaissance/Probing',
            'Analysis': 'Analysis Attacks',
            'Backdoors': 'Backdoor Attacks',
            'Shellcode': 'Shellcode Attacks',
            'Worms': 'Worm Attacks'
        }
    
    def load_data(self):
        """Load and merge UNSW-NB15 numbered files for analysis"""
        print("🔄 Loading UNSW-NB15 dataset for analysis...")
        
        numbered_files = [
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv", 
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv"
        ]
        
        dataframes = []
        
        for filename in numbered_files:
            file_path = self.dataset_path / filename
            if file_path.exists():
                print(f"   📁 Loading {filename}...")
                try:
                    df = pd.read_csv(file_path, header=None, low_memory=False)
                    
                    # Assign column names
                    if len(df.columns) <= len(self.column_names):
                        df.columns = self.column_names[:len(df.columns)]
                    else:
                        df.columns = self.column_names + [f'extra_col_{i}' for i in range(len(df.columns) - len(self.column_names))]
                    
                    df['source_file'] = filename
                    dataframes.append(df)
                    print(f"      ✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    print(f"   ❌ Error loading {filename}: {e}")
        
        if not dataframes:
            raise FileNotFoundError("No UNSW-NB15 numbered files found!")
        
        # Merge for analysis (not saved)
        print(f"🔄 Merging {len(dataframes)} files for analysis...")
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Merged dataset: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
        
        # Clean data
        self.data.columns = self.data.columns.str.strip()
        
    def analyze_unique_values(self):
        """Analyze unique values for all columns"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE UNIQUE VALUES ANALYSIS")
        print("="*80)
        
        total_rows = len(self.data)
        
        for col in self.data.columns:
            if col == 'source_file':  # Skip metadata column
                continue
                
            print(f"\n🔍 Column: {col}")
            print("-" * 60)
            
            # Basic stats
            unique_count = self.data[col].nunique()
            null_count = self.data[col].isnull().sum()
            null_percent = (null_count / total_rows) * 100
            
            print(f"   Total unique values: {unique_count:,}")
            print(f"   Null/Missing values: {null_count:,} ({null_percent:.2f}%)")
            print(f"   Data type: {self.data[col].dtype}")
            
            # Handle different column types
            if self.data[col].dtype == 'object' or unique_count <= 50:
                # Show all unique values for categorical or small unique sets
                value_counts = self.data[col].value_counts(dropna=False)
                print(f"   \n   📋 All unique values ({len(value_counts)}):")
                
                for i, (value, count) in enumerate(value_counts.items()):
                    percentage = (count / total_rows) * 100
                    value_str = str(value) if pd.notna(value) else "NULL/NaN"
                    print(f"      {i+1:2d}. {value_str:<20} | {count:>10,} ({percentage:>6.2f}%)")
                    
            elif self.data[col].dtype in ['int64', 'float64']:
                # For numeric columns with many unique values
                print(f"   📊 Numeric Statistics:")
                stats = self.data[col].describe()
                print(f"      Mean: {stats['mean']:.6f}")
                print(f"      Std:  {stats['std']:.6f}")
                print(f"      Min:  {stats['min']:.6f}")
                print(f"      25%:  {stats['25%']:.6f}")
                print(f"      50%:  {stats['50%']:.6f}")
                print(f"      75%:  {stats['75%']:.6f}")
                print(f"      Max:  {stats['max']:.6f}")
                
                # Show top 20 most frequent values
                value_counts = self.data[col].value_counts().head(20)
                if len(value_counts) > 0:
                    print(f"   \n   🔝 Top 20 most frequent values:")
                    for i, (value, count) in enumerate(value_counts.items()):
                        percentage = (count / total_rows) * 100
                        print(f"      {i+1:2d}. {value:<20} | {count:>10,} ({percentage:>6.2f}%)")
            
            else:
                # For other data types, show top 20 values
                value_counts = self.data[col].value_counts().head(20)
                print(f"   \n   🔝 Top 20 most frequent values:")
                for i, (value, count) in enumerate(value_counts.items()):
                    percentage = (count / total_rows) * 100
                    value_str = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                    print(f"      {i+1:2d}. {value_str:<20} | {count:>10,} ({percentage:>6.2f}%)")
    
    def analyze_attack_distribution(self):
        """Comprehensive attack distribution analysis"""
        print("\n" + "="*80)
        print("🎯 ATTACK DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Find attack columns
        attack_col = None
        label_col = None
        
        for col in ['attack_cat', 'Attack_cat', 'attack_category']:
            if col in self.data.columns:
                attack_col = col
                break
        
        for col in ['label', 'Label']:
            if col in self.data.columns:
                label_col = col
                break
        
        if attack_col:
            print(f"✅ Using attack category column: '{attack_col}'")
            
            # Clean attack categories
            self.data[attack_col] = self.data[attack_col].astype(str).str.strip()
            self.data[attack_col] = self.data[attack_col].replace(['', 'nan', 'NaN'], 'Normal')
            
            attack_counts = self.data[attack_col].value_counts()
            total_records = len(self.data)
            
            print(f"\n📊 Attack Category Distribution:")
            print("-" * 80)
            print(f"{'Attack Type':<25} | {'Count':<12} | {'Percentage':<12} | {'Description'}")
            print("-" * 80)
            
            for attack, count in attack_counts.items():
                percentage = (count / total_records) * 100
                description = self.attack_descriptions.get(attack, 'Unknown Attack Type')
                print(f"{attack:<25} | {count:>10,} | {percentage:>9.2f}% | {description}")
            
            # Attack vs Normal ratio
            normal_count = attack_counts.get('Normal', 0)
            attack_total = total_records - normal_count
            
            print(f"\n🏷️  Binary Classification Summary:")
            print("-" * 50)
            print(f"Normal Traffic:  {normal_count:>10,} ({(normal_count/total_records)*100:>6.2f}%)")
            print(f"Attack Traffic:  {attack_total:>10,} ({(attack_total/total_records)*100:>6.2f}%)")
            print(f"Attack Ratio:    1 attack for every {normal_count/attack_total if attack_total > 0 else 0:.1f} normal flows")
        
        if label_col:
            print(f"\n✅ Binary label column: '{label_col}'")
            
            # Analyze binary labels
            self.data[label_col] = pd.to_numeric(self.data[label_col], errors='coerce')
            label_counts = self.data[label_col].value_counts()
            
            print(f"\n🔢 Binary Label Distribution:")
            print("-" * 40)
            for label, count in label_counts.items():
                percentage = (count / len(self.data)) * 100
                label_name = "Normal" if label == 0 else "Attack"
                print(f"{label_name} ({label}): {count:>10,} ({percentage:>6.2f}%)")
    
    def analyze_cross_correlations(self):
        """Analyze relationships between key categorical features"""
        print("\n" + "="*80)
        print("🔗 CROSS-FEATURE ANALYSIS")
        print("="*80)
        
        # Key categorical features to analyze
        key_features = ['proto', 'state', 'service', 'attack_cat']
        available_features = [f for f in key_features if f in self.data.columns]
        
        if len(available_features) >= 2:
            print(f"📊 Analyzing relationships between: {', '.join(available_features)}")
            
            # Protocol vs Attack
            if 'proto' in available_features and 'attack_cat' in available_features:
                print(f"\n🔌 Protocol vs Attack Analysis:")
                print("-" * 60)
                crosstab = pd.crosstab(self.data['proto'], self.data['attack_cat'])
                
                # Show top protocol-attack combinations
                for proto in self.data['proto'].value_counts().head(5).index:
                    for attack in self.data['attack_cat'].value_counts().head(5).index:
                        if proto in crosstab.index and attack in crosstab.columns:
                            count = crosstab.loc[proto, attack]
                            if count > 0:
                                percentage = (count / len(self.data)) * 100
                                print(f"   {proto:<10} + {attack:<15}: {count:>8,} ({percentage:>5.2f}%)")
            
            # State vs Attack
            if 'state' in available_features and 'attack_cat' in available_features:
                print(f"\n🔄 Connection State vs Attack Analysis:")
                print("-" * 60)
                crosstab = pd.crosstab(self.data['state'], self.data['attack_cat'])
                
                for state in self.data['state'].value_counts().head(5).index:
                    for attack in self.data['attack_cat'].value_counts().head(5).index:
                        if state in crosstab.index and attack in crosstab.columns:
                            count = crosstab.loc[state, attack]
                            if count > 0:
                                percentage = (count / len(self.data)) * 100
                                print(f"   {state:<10} + {attack:<15}: {count:>8,} ({percentage:>5.2f}%)")
    
    def run_analysis(self):
        """Run complete unique values and attack distribution analysis"""
        print("🚀 UNSW-NB15 Unique Values & Attack Distribution Analysis")
        print("=" * 80)
        print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Load data (merge for analysis only)
            self.load_data()
            
            # Analyze unique values for all columns
            self.analyze_unique_values()
            
            # Attack distribution analysis
            self.analyze_attack_distribution()
            
            # Cross-feature analysis
            self.analyze_cross_correlations()
            
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETED!")
            print("="*80)
            print(f"📅 Analysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 Analyzed {len(self.data):,} records across {len(self.data.columns)} features")
            print("📁 Results saved to eda_outputs directory")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution function with output capture"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"UNSW_NB15_Unique_Analysis_{timestamp}.txt"
    
    # Capture all output
    output_capture = OutputCapture(output_filename)
    sys.stdout = output_capture
    
    try:
        print("UNSW-NB15 Unique Values & Attack Distribution Analysis")
        print("=" * 60)
        
        # Initialize analyzer
        analyzer = UNSWUniqueAnalyzer()
        
        # Run analysis
        analyzer.run_analysis()
        
    finally:
        # Restore stdout and save output
        sys.stdout = output_capture.terminal
        output_capture.save_to_file()
        print(f"\n✅ Complete analysis saved to: eda_outputs/{output_filename}")

if __name__ == "__main__":
    main()
