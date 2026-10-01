import os
import shutil
from datetime import datetime

# 1. Define your original files and folders
ORIGINAL_DATASET = "./e-waste visual dataset"
ORIGINAL_EXCEL = "your_spreadsheet.xlsx"  # Change this to your actual Excel filename

# 2. Generate a unique name using current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_FOLDER = f"./BACKUP_e_waste_{timestamp}"

print("🔄 Starting data backup...")

try:
    # 3. Create the backup main directory
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    
    # 4. Backup the Excel spreadsheet if it exists
    if os.path.exists(ORIGINAL_EXCEL):
        shutil.copy2(ORIGINAL_EXCEL, os.path.join(BACKUP_FOLDER, os.path.basename(ORIGINAL_EXCEL)))
        print(f"✅ Excel sheet backed up successfully to: {BACKUP_FOLDER}")
    else:
        print("⚠️ Warning: Original Excel file not found. Skipping Excel backup.")
    
    # 5. Backup the entire dataset folder if it exists
    if os.path.exists(ORIGINAL_DATASET):
        dest_dataset_path = os.path.join(BACKUP_FOLDER, os.path.basename(ORIGINAL_DATASET))
        shutil.copytree(ORIGINAL_DATASET, dest_dataset_path)
        print(f"✅ Image dataset backed up successfully to: {dest_dataset_path}")
    else:
        print("⚠️ Warning: Original image dataset folder not found. Skipping image backup.")
    
    print(f"\n🎉 Success! All original data is safe inside the folder: {BACKUP_FOLDER}")

except Exception as e:
    print(f"❌ Backup failed due to an error: {e}")