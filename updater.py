import os
import shutil
import pandas as pd
from pathlib import Path
from icrawler.builtin import BingImageCrawler

# 1. Configuration
EXCEL_FILE = str(Path(__file__).resolve().parent / "emadini" / "e-waste datasets.xlsx")
DATASET_DIR = "./e-waste visual dataset"
IMAGES_PER_DEVICE = 5  # Number of images to download per device

print("🔄 Initializing e-waste dataset image updater...")

# 2. Load the Excel file safely
try:
    df = pd.read_excel(EXCEL_FILE)
    print(f"📊 Successfully loaded Excel file: '{EXCEL_FILE}' with {len(df)} rows.")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{EXCEL_FILE}' in this directory.")
    print("Please verify the filename and try again.")
    exit()

# Function to sanitize folder names
def sanitize_folder_name(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()

# 3. Process each row
for index, row in df.iterrows():
    # Extract search query from the first column
    search_query = str(row.iloc[0]).strip()
    
    # Skip empty or invalid entries
    if not search_query or search_query.lower() == 'nan':
        print(f"⚠️ Row {index + 2}: Empty or invalid search term. Skipping.")
        continue

    # Create a clean folder name
    folder_name = sanitize_folder_name(search_query)
    target_folder = os.path.join(DATASET_DIR, folder_name)

    print(f"\n──────────────────────────────────────────────────")
    print(f"📦 Row {index + 2}: Processing '{search_query}'")
    
    # Remove existing folder to avoid mixing images
    if os.path.exists(target_folder):
        print(f"🗑️  Removing old folder: {folder_name}")
        shutil.rmtree(target_folder)
        
    os.makedirs(target_folder, exist_ok=True)
    
    # Download images
    print(f"🚀 Downloading {IMAGES_PER_DEVICE} images for '{search_query}'...")
    try:
        crawler = BingImageCrawler(downloader_threads=2, storage={'root_dir': target_folder})
        crawler.crawl(keyword=search_query, max_num=IMAGES_PER_DEVICE)
    except Exception as e:
        print(f"⚠️  Error during image download for '{search_query}': {e}")

print("\n🎉 Done! All folders have been updated with precise electronic device images.")