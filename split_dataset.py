import os
import random
import shutil

# Paths
SOURCE_DIR = "./MASTER_e_waste_visual_dataset"
TRAIN_DIR = "./dataset_split/train"
VAL_DIR = "./dataset_split/val"

# Split ratio (80% train, 20% validation)
SPLIT_RATIO = 0.8

print("📦 Starting to split your dataset into Train and Validation sets...")

# Check if source directory exists
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Error: Could not find folder '{SOURCE_DIR}'")
    exit()

# Loop through each e-waste device folder
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)
    
    if os.path.isdir(folder_path):
        # List all image files
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Skip if no images found
        if len(images) == 0:
            continue
            
        # Shuffle images randomly
        random.shuffle(images)
        
        # Determine split index
        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create target directories if they don't exist
        train_folder_path = os.path.join(TRAIN_DIR, folder)
        val_folder_path = os.path.join(VAL_DIR, folder)
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(val_folder_path, exist_ok=True)
        
        # Copy images to training folder
        for img in train_images:
            src_path = os.path.join(folder_path, img)
            dst_path = os.path.join(train_folder_path, img)
            shutil.copy2(src_path, dst_path)
        
        # Copy images to validation folder
        for img in val_images:
            src_path = os.path.join(folder_path, img)
            dst_path = os.path.join(val_folder_path, img)
            shutil.copy2(src_path, dst_path)

print("\n🎉 Success! Your images are split into training and validation sets.")
print(f"📁 Training folder: {TRAIN_DIR}")
print(f"📁 Validation folder: {VAL_DIR}")