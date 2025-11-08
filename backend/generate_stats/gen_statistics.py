import pandas as pd
import numpy as np
import json

def generate_traffic_statistics(csv_file_path, output_json_path='traffic_statistics.json'):
    """
    Generate comprehensive statistics for network traffic dataset.

    Parameters:
    csv_file_path (str): Path to the CSV file
    output_json_path (str): Path for the output JSON file
    """

    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Initialize statistics dictionary
    stats = {}

    # Basic counts
    stats['number_of_data_points'] = int(len(df))
    stats['anomaly_count'] = int((df['label'] == 1).sum())
    stats['benign_count'] = int((df['label'] == 0).sum())

    # Calculate data size in bytes and convert to human-readable format
    data_size_bytes = df.memory_usage(deep=True).sum()
    stats['data_size'] = {
        'bytes': int(data_size_bytes),
        'kilobytes': round(data_size_bytes / 1024, 2),
        'megabytes': round(data_size_bytes / (1024 ** 2), 2),
        'gigabytes': round(data_size_bytes / (1024 ** 3), 2)
    }

    # Add traffic class labels
    df['traffic_class'] = df['label'].map({0: 'BENIGN', 1: 'ANOMALY'})

    # Features to analyze
    features = ['duration', 'bytes_per_sec', 'avg_pkt_size', 'pkts_per_sec']

    # Box plot data - distribution by traffic class
    stats['box_plot_data'] = {}
    for feature in features:
        stats['box_plot_data'][feature] = {}
        for label in ['BENIGN', 'ANOMALY']:
            data = df[df['traffic_class'] == label][feature].dropna()
            stats['box_plot_data'][feature][label] = {
                'min': float(data.min()) if len(data) > 0 else None,
                'q1': float(data.quantile(0.25)) if len(data) > 0 else None,
                'median': float(data.median()) if len(data) > 0 else None,
                'q3': float(data.quantile(0.75)) if len(data) > 0 else None,
                'max': float(data.max()) if len(data) > 0 else None,
                'mean': float(data.mean()) if len(data) > 0 else None,
                'std': float(data.std()) if len(data) > 0 else None,
                'count': int(len(data))
            }

    # Scatter plot data - Packet size vs bytes per second
    stats['scatter_plot_data'] = {
        'packet_size_vs_bytes_per_sec': {}
    }
    for label in ['BENIGN', 'ANOMALY']:
        subset = df[df['traffic_class'] == label][['avg_pkt_size', 'bytes_per_sec']].dropna()
        # Sample if dataset is too large (limit to 1000 points per class for visualization)
        if len(subset) > 1000:
            subset = subset.sample(n=1000, random_state=42)
        stats['scatter_plot_data']['packet_size_vs_bytes_per_sec'][label] = {
            'x': subset['avg_pkt_size'].tolist(),
            'y': subset['bytes_per_sec'].tolist(),
            'count': int(len(subset))
        }

    # Histogram data - distribution by traffic class
    stats['histogram_data'] = {}
    for feature in features:
        stats['histogram_data'][feature] = {}
        for label in ['BENIGN', 'ANOMALY']:
            data = df[df['traffic_class'] == label][feature].dropna()
            if len(data) > 0:
                # Create histogram with 30 bins
                counts, bin_edges = np.histogram(data, bins=30)
                stats['histogram_data'][feature][label] = {
                    'counts': counts.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
                    'total_count': int(len(data))
                }
            else:
                stats['histogram_data'][feature][label] = {
                    'counts': [],
                    'bin_edges': [],
                    'bin_centers': [],
                    'total_count': 0
                }

    # Additional summary statistics
    stats['summary_statistics'] = {}
    for feature in features:
        stats['summary_statistics'][feature] = {
            'overall': {
                'mean': float(df[feature].mean()),
                'median': float(df[feature].median()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max())
            }
        }
        for label in ['BENIGN', 'ANOMALY']:
            data = df[df['traffic_class'] == label][feature]
            stats['summary_statistics'][feature][label] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max())
            }

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics generated successfully!")
    print(f"Output saved to: {output_json_path}")
    print(f"\nSummary:")
    print(f"Total data points: {stats['number_of_data_points']}")
    print(f"Benign traffic: {stats['benign_count']} ({stats['benign_count']/stats['number_of_data_points']*100:.2f}%)")
    print(f"Anomaly traffic: {stats['anomaly_count']} ({stats['anomaly_count']/stats['number_of_data_points']*100:.2f}%)")

    return stats

# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual file path
    csv_file = 'dataset.csv'
    output_file = 'traffic_statistics.json'

    try:
        statistics = generate_traffic_statistics(csv_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please update the 'csv_file' variable with the correct path to your dataset.")
    except Exception as e:
        print(f"Error: {e}")