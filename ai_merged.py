import os
import shutil
import pandas as pd
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# --- Configuration & Paths ---      
EXCEL_1 = str(Path(__file__).resolve().parent / "emadini" / "e-waste datasets.xlsx")
# You mentioned only merging one Excel, so comment out or remove the second Excel if not used:
# EXCEL_2 = "updated_e_waste_datasets.xlsx"
DIR_1 = "./e-waste visual dataset"
# If only one dataset, ignore the second directory:
# DIR_2 = "./updated e-waste visual dataset"
MASTER_EXCEL = "MASTER_e_waste_dataset.xlsx"
MASTER_DIR = "./MASTER_e_waste_visual_dataset"

# Ensure master directory exists
os.makedirs(MASTER_DIR, exist_ok=True)

# --- Load Pre-trained Model ---
print("🧠 Loading AI Computer Vision Model...")
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)
model.eval()
preprocess = weights.transforms()

# Define categories related to electronic devices (ImageNet class IDs)
ELECTRONIC_CATEGORIES = {
    527, 620, 725, 487, 610, 664, 852, 704, 508, 684, 826, 755, 718,
    542, 652, 414, 513, 811, 794, 631, 881, 401, 804, 730
}  # Example: mouse, keyboard, laptop, phone, etc.

def is_valid_electronic(image_path):
    """Determine if the image contains an electronic device using AI."""
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        topk_probs, topk_classes = torch.topk(probs, 3)
        for class_id in topk_classes:
            if class_id.item() in ELECTRONIC_CATEGORIES:
                return True
        return False
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

# --- Load and Merge Excel Sheets ---
print("📊 Merging Excel records...")
if not os.path.exists(EXCEL_1):
    print(f"❌ Error: Excel file '{EXCEL_1}' is missing.")
    exit()

df1 = pd.read_excel(EXCEL_1)

# If only one Excel file, skip merging with a second file
# If you want to include a second file, uncomment below:
# if not os.path.exists(EXCEL_2):
#     print(f"❌ Error: Excel file '{EXCEL_2}' is missing.")
#     exit()
# df2 = pd.read_excel(EXCEL_2)
# combined_df = pd.concat([df1, df2], ignore_index=True)
# combined_df = combined_df.drop_duplicates(subset=[combined_df.columns[0]], keep='first')

# Since only one Excel, just assign df1
combined_df = df1

# --- Review and Filter Images ---
print("📸 Reviewing images with AI and merging folders...")
all_folders = set()
if os.path.exists(DIR_1):
    all_folders.update(os.listdir(DIR_1))
# If only one directory, skip the second
# if os.path.exists(DIR_2):
#     all_folders.update(os.listdir(DIR_2))

valid_device_names = []

# Loop through each folder (dataset)
for folder in all_folders:
    target_folder = os.path.join(MASTER_DIR, folder)
    image_count = 0
    src_dirs = [os.path.join(DIR_1, folder)]
    # If using second directory, add:
    # src_dirs.append(os.path.join(DIR_2, folder))
    
    for src_dir in src_dirs:
        if os.path.isdir(src_dir):
            for file_name in os.listdir(src_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(src_dir, file_name)
                    if is_valid_electronic(file_path):
                        os.makedirs(target_folder, exist_ok=True)
                        new_name = f"elec_{image_count:04d}.jpg"
                        shutil.copy2(file_path, os.path.join(target_folder, new_name))
                        image_count += 1
                    else:
                        print(f"🗑️ Filtered out: {file_path}")
    
    if image_count > 0:
        valid_device_names.append(folder)
    else:
        # Remove empty folders if they exist
        if os.path.exists(target_folder):
            try:
                os.rmdir(target_folder)
            except OSError:
                pass  # Directory not empty or other error

# --- Update Excel Data ---
print("📝 Filtering dataset based on validated electronic devices...")
master_rows = []

for _, row in combined_df.iterrows():
    raw_name = str(row.iloc[0]).strip()
    sanitized_name = "".join(c for c in raw_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if sanitized_name in valid_device_names:
        master_rows.append(row)

if master_rows:
    final_df = pd.DataFrame(master_rows)
    final_df.to_excel(MASTER_EXCEL, index=False)
    print(f"\n🎉 Process completed! Master dataset saved at '{MASTER_EXCEL}'.")
    print(f"📊 {len(final_df)} entries retained.")
else:
    print("\n⚠️ No entries passed the electronic device filter.")