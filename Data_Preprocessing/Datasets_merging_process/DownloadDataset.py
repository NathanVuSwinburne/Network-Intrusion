import kagglehub
import os
import shutil
from pathlib import Path

def download_dataset_to_path(dataset_name, target_path):
    """
    Download a Kaggle dataset to a specific path, avoiding cache directory.
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        target_path (str): Target directory path for the dataset
    """
    print(f"Downloading {dataset_name}...")
    
    # Create target directory if it doesn't exist
    Path(target_path).mkdir(parents=True, exist_ok=True)
    
    # Download to default cache location first
    cache_path = kagglehub.dataset_download(dataset_name)
    print(f"Downloaded to cache: {cache_path}")
    
    # Copy files from cache to target directory
    if os.path.exists(cache_path):
        # Copy all files and subdirectories
        for item in os.listdir(cache_path):
            source_item = os.path.join(cache_path, item)
            target_item = os.path.join(target_path, item)
            
            if os.path.isdir(source_item):
                # Copy directory recursively
                if os.path.exists(target_item):
                    shutil.rmtree(target_item)
                shutil.copytree(source_item, target_item)
            else:
                # Copy file
                shutil.copy2(source_item, target_item)
        
        print(f"Dataset copied to: {target_path}")
        
        # Optional: Clean up cache directory to save space
        try:
            shutil.rmtree(cache_path)
            print(f"Cleaned up cache directory: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not clean up cache directory: {e}")
    else:
        print(f"Error: Cache path not found: {cache_path}")

def main():
    """Main function to download both datasets."""
    print("Starting dataset downloads...")
    
    # Define dataset configurations
    datasets = [
        {
            "name": "mrwellsdavid/unsw-nb15",
            "path": "network-intrusion-dataset/UNSW_NB15",
            "description": "UNSW-NB15 Dataset"
        },
        {
            "name": "chethuhn/network-intrusion-dataset", 
            "path": "network-intrusion-dataset/CIC_IDS_2017",
            "description": "CIC-IDS-2017 Dataset"
        }
    ]
    
    # Download each dataset
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {dataset['description']}")
        print(f"{'='*50}")
        
        try:
            download_dataset_to_path(dataset["name"], dataset["path"])
            print(f" Successfully downloaded {dataset['description']} to {dataset['path']}")
        except Exception as e:
            print(f" Error downloading {dataset['description']}: {e}")
    
    print(f"\n{'='*50}")
    print("Dataset download process completed!")
    print(f"{'='*50}")
    
    # Display final directory structure
    print("\nFinal directory structure:")
    base_path = "network-intrusion-dataset"
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    main()