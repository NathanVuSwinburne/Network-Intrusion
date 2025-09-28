#!/usr/bin/env python3
"""
UNSW-NB15 Focused Analysis: Protocol, State, Label, Destination Port
This script analyzes unique values for only the specified columns and provides 
comprehensive duplication statistics for the dataset.
Results are saved to EDA/Outputs directory.
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
        output_dir = Path("EDA/Outputs")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / self.filename, 'w', encoding='utf-8') as f:
            f.write(self.log.getvalue())

class UNSWFocusedAnalyzer:
    """
    Focused analyzer for UNSW-NB15 dataset - Protocol, State, attack_cat, destination port only
    """
    
    def __init__(self, dataset_path="data/network-intrusion-dataset/UNSW_NB15"):
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
        
        # Focus columns only - exactly what user requested
        self.focus_columns = ['proto', 'state', 'attack_cat','dsport']
        
    def load_data(self):
        """Load and merge UNSW-NB15 numbered files for analysis"""
        print("🔄 Loading UNSW-NB15 dataset for focused analysis...")
        
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
        
        # Merge for analysis
        print(f"🔄 Merging {len(dataframes)} files for analysis...")
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Merged dataset: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
        
        # Clean data
        self.data.columns = self.data.columns.str.strip()
        
    def analyze_duplication(self):
        """Comprehensive duplication analysis"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DUPLICATION ANALYSIS")
        print("="*80)
        
        total_rows = len(self.data)
        
        # 1. Complete row duplicates (all columns)
        print("\n📊 1. COMPLETE ROW DUPLICATES (All Columns)")
        print("-" * 60)
        
        # Exclude source_file column for duplication check
        analysis_data = self.data.drop(columns=['source_file'], errors='ignore')
        complete_duplicates = analysis_data.duplicated()
        complete_duplicate_count = complete_duplicates.sum()
        unique_complete_rows = total_rows - complete_duplicate_count
        
        print(f"Total rows:           {total_rows:,}")
        print(f"Unique rows:          {unique_complete_rows:,}")
        print(f"Duplicate rows:       {complete_duplicate_count:,}")
        print(f"Duplication rate:     {(complete_duplicate_count/total_rows)*100:.2f}%")
        print(f"Uniqueness rate:      {(unique_complete_rows/total_rows)*100:.2f}%")
        
        # 2. Focus columns duplicates
        print(f"\n📊 2. FOCUS COLUMNS DUPLICATES ({', '.join(self.focus_columns)})")
        print("-" * 60)
        
        focus_data = self.data[self.focus_columns].copy()
        focus_duplicates = focus_data.duplicated()
        focus_duplicate_count = focus_duplicates.sum()
        unique_focus_rows = total_rows - focus_duplicate_count
        
        print(f"Total rows:           {total_rows:,}")
        print(f"Unique combinations:  {unique_focus_rows:,}")
        print(f"Duplicate combinations: {focus_duplicate_count:,}")
        print(f"Duplication rate:     {(focus_duplicate_count/total_rows)*100:.2f}%")
        print(f"Uniqueness rate:      {(unique_focus_rows/total_rows)*100:.2f}%")
        
        # 3. Individual column duplicates
        print(f"\n📊 3. INDIVIDUAL COLUMN DUPLICATION RATES")
        print("-" * 60)
        
        for col in self.focus_columns:
            if col in self.data.columns:
                unique_values = self.data[col].nunique()
                duplication_rate = ((total_rows - unique_values) / total_rows) * 100
                print(f"{col:<15}: {unique_values:>8,} unique | {duplication_rate:>6.2f}% duplication")
        
        # 4. Most duplicated combinations
        print(f"\n📊 4. TOP 20 MOST DUPLICATED FOCUS COMBINATIONS")
        print("-" * 80)
        
        focus_value_counts = focus_data.value_counts()
        duplicated_combinations = focus_value_counts[focus_value_counts > 1].head(20)
        
        if len(duplicated_combinations) > 0:
            print(f"{'Rank':<4} | {'Protocol':<8} | {'State':<10} | {'Attack_cat':<6} | {'Port':<8} | {'Count':<10} | {'%'}")
            print("-" * 80)
            
            for i, (combination, count) in enumerate(duplicated_combinations.items(), 1):
                proto, state, attack_cat, port = combination
                percentage = (count / total_rows) * 100
                print(f"{i:<4} | {str(proto):<8} | {str(state):<10} | {str(attack_cat):<6} | {str(port):<8} | {count:<10,} | {percentage:.2f}%")
        else:
            print("No duplicated combinations found in focus columns.")
        
        # 5. Duplication by source file
        print(f"\n📊 5. DUPLICATION ANALYSIS BY SOURCE FILE")
        print("-" * 60)
        
        if 'source_file' in self.data.columns:
            for filename in self.data['source_file'].unique():
                file_data = self.data[self.data['source_file'] == filename]
                file_rows = len(file_data)
                
                # Complete duplicates within file
                file_analysis_data = file_data.drop(columns=['source_file'], errors='ignore')
                file_complete_dups = file_analysis_data.duplicated().sum()
                
                # Focus duplicates within file
                file_focus_data = file_data[self.focus_columns]
                file_focus_dups = file_focus_data.duplicated().sum()
                
                print(f"{filename}:")
                print(f"   Rows: {file_rows:,}")
                print(f"   Complete duplicates: {file_complete_dups:,} ({(file_complete_dups/file_rows)*100:.2f}%)")
                print(f"   Focus duplicates: {file_focus_dups:,} ({(file_focus_dups/file_rows)*100:.2f}%)")
        
    def analyze_focus_unique_values(self):
        """Analyze unique values for focus columns only"""
        print("\n" + "="*80)
        print("📊 FOCUS COLUMNS UNIQUE VALUES ANALYSIS")
        print("="*80)
        print(f"Focus columns: {', '.join(self.focus_columns)}")
        
        total_rows = len(self.data)
        
        for col in self.focus_columns:
            if col not in self.data.columns:
                print(f"\n❌ Column '{col}' not found in dataset!")
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
            
            # Show all unique values with counts and percentages
            value_counts = self.data[col].value_counts(dropna=False)
            print(f"   \n   📋 All unique values ({len(value_counts)}):")
            # Special handling for dsport - only show ports with >1%
            if col == 'dsport':
                # Filter for ports with >1% occurrence
                significant_ports = value_counts[value_counts / total_rows > 0.01]
                other_ports_count = value_counts[value_counts / total_rows <= 0.01].sum()
                for i, (value, count) in enumerate(significant_ports.items()):
                    percentage = (count / total_rows) * 100
                    value_str = str(value) if pd.notna(value) else "NULL/NaN"
                    print(f"      {i+1:2d}. {value_str:<20} | {count:>10,} ({percentage:>6.2f}%)")
                # Show summary of other ports
                if other_ports_count > 0:
                    other_percentage = (other_ports_count / total_rows) * 100
                    other_unique_count = len(value_counts[value_counts / total_rows <= 0.01])
                    print(f"      ... {other_unique_count:,} other ports (<1% each) | {other_ports_count:>10,} ({other_percentage:>6.2f}%)")
            else:
                # For other columns, show all values as before
                print(f"   \n   📋 All unique values ({len(value_counts)}):")
                for i, (value, count) in enumerate(value_counts.items()):
                    percentage = (count / total_rows) * 100
                    value_str = str(value) if pd.notna(value) else "NULL/NaN"
                    print(f"      {i+1:2d}. {value_str:<20} | {count:>10,} ({percentage:>6.2f}%)")
    
    def run_analysis(self):
        """Run complete focused analysis"""
        print("🚀 UNSW-NB15 Focused Analysis: Protocol, State, attack_cat, Destination Port")
        print("=" * 80)
        print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Load data
            self.load_data()
            
            # Duplication analysis (comprehensive)
            self.analyze_duplication()
            
            # Focus columns unique values
            self.analyze_focus_unique_values()
            
            print("\n" + "="*80)
            print("✅ FOCUSED ANALYSIS COMPLETED!")
            print("="*80)
            print(f"📅 Analysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 Analyzed {len(self.data):,} records")
            print(f"🎯 Focus columns: {', '.join(self.focus_columns)}")
            print("📁 Results saved to EDA/Outputs directory")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution function with output capture"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"UNSW_NB15_Focused_Analysis_{timestamp}.txt"
    
    # Capture all output
    output_capture = OutputCapture(output_filename)
    sys.stdout = output_capture
    
    try:
        print("UNSW-NB15 Focused Analysis: Protocol, State, attack_cat, Destination Port")
        print("=" * 70)
        
        # Initialize analyzer
        analyzer = UNSWFocusedAnalyzer()
        
        # Run analysis
        analyzer.run_analysis()
        
    finally:
        # Restore stdout and save output
        sys.stdout = output_capture.terminal
        output_capture.save_to_file()
        print(f"\n✅ Complete focused analysis saved to: EDA/Outputs/{output_filename}")

if __name__ == "__main__":
    main()
