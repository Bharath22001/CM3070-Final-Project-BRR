import os
import glob
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration: the folder holding all the bbox annotation CSVs
ANNOTATIONS_DIR = "cuneiform_data/cuneiform-sign-detection-dataset/annotations"
IMG_DIR = "cuneiform_data/images"

def get_all_missing_tablets(annotations_dir, img_dir):
    all_tablets = set()
    
    # Find every bbox_annotations_*.csv file
    search_pattern = os.path.join(annotations_dir, "bbox_annotations_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching {search_pattern}")
        return [], 0

    print(f"Scanning {len(csv_files)} CSV files for unique tablet identifiers...")
    
    # Collect the P-numbers from each CSV
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            if 'tablet_CDLI' in df.columns:
                all_tablets.update(df['tablet_CDLI'].unique())
        except Exception as e:
            print(f"Could not read {os.path.basename(path)}: {e}")
    
    # Keep only the tablets I don't already have an image for
    missing = [t for t in all_tablets if not os.path.exists(os.path.join(img_dir, f"{t}.jpg"))]
    return missing, len(all_tablets)

def create_robust_session():
    """A requests session that retries automatically when the CDLI servers fail."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,  # waits 1, 2, 4, 8 s between retries
        status_forcelist=(500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def scrape_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    
    missing, total_unique = get_all_missing_tablets(ANNOTATIONS_DIR, IMG_DIR)
    
    print("-" * 50)
    print(f"Total unique tablets across all CSVs: {total_unique}")
    print(f"Tablets currently missing (to be downloaded): {len(missing)}")
    print("-" * 50)
    
    if len(missing) == 0:
        print("Your dataset is completely synced! All images are present in the directory.")
        return

    session = create_robust_session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    success_count = 0
    
    for i, tablet_id in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] Fetching {tablet_id:<10} ...", end=" ", flush=True)
        
        url_primary = f"https://cdli.mpiwg-berlin.mpg.de/dl/photo/{tablet_id}.jpg"
        url_fallback = f"https://cdli.ucla.edu/dl/photo/{tablet_id}.jpg" 
        
        try:
            time.sleep(0.5) # polite delay so I don't get blocked
            
            # Try the main Berlin server first
            resp = session.get(url_primary, headers=headers, stream=True, timeout=15)
            
            if resp.status_code == 200:
                with open(os.path.join(IMG_DIR, f"{tablet_id}.jpg"), 'wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                print("SUCCESS (Berlin)")
                success_count += 1
                continue
                
            # Then the UCLA mirror
            resp_fallback = session.get(url_fallback, headers=headers, stream=True, timeout=15)
            
            if resp_fallback.status_code == 200:
                with open(os.path.join(IMG_DIR, f"{tablet_id}.jpg"), 'wb') as f:
                    for chunk in resp_fallback.iter_content(8192):
                        f.write(chunk)
                print("SUCCESS (UCLA)")
                success_count += 1
            else:
                print(f"FAILED (Not on Berlin [HTTP {resp.status_code}] or UCLA [HTTP {resp_fallback.status_code}])")
                
        except Exception as e:
            print(f"FAILED (Network/System Error)")

    print(f"\nDownload complete! Added {success_count} new physical tablet images.")

if __name__ == "__main__":
    scrape_images()