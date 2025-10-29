#!/usr/bin/env python3
"""
Download arXiv metadata from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the kaggle.json file to ~/.kaggle/
   - chmod 600 ~/.kaggle/kaggle.json

Usage:
    python download_arxiv_metadata.py
"""

import shutil
from pathlib import Path

def download_arxiv_metadata():
    """Download and extract arXiv metadata from Kaggle."""
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Please run: pip install kaggle")
        return
    
    # Check if credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Error: Kaggle credentials not found.")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save the kaggle.json file to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Create data directory
    data_dir = Path('json2bibtex/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading arXiv metadata from Kaggle...")
    print("This is a 3.5GB file and may take a while...")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            'Cornell-University/arxiv',
            path=str(data_dir),
            unzip=True
        )
        
        print("Download complete! Extracting files...")
        
        # The file might be named differently, let's check
        json_files = list(data_dir.glob('*.json'))
        if json_files:
            # Rename to expected filename
            source_file = json_files[0]
            target_file = data_dir / 'arxiv-metadata-oai-snapshot.json'
            
            if source_file != target_file:
                shutil.move(str(source_file), str(target_file))
            
            print(f"Success! File saved to: {target_file}")
        else:
            print("Error: No JSON file found after extraction.")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://www.kaggle.com/datasets/Cornell-University/arxiv")
        print("Then extract the JSON file to: json2bibtex/data/")


if __name__ == "__main__":
    download_arxiv_metadata()