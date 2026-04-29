import os
import hashlib
from collections import defaultdict

# 🔹 Generate hash of image
def get_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

# 🔹 Find duplicates across folders
def check_duplicates(dataset_path):
    hash_dict = defaultdict(list)

    print("🔍 Checking images...\n")

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(root, file)
                img_hash = get_hash(full_path)

                if img_hash:
                    hash_dict[img_hash].append(full_path)

    # 🔹 Show duplicates (only across different folders)
    found = False
    total_duplicates = 0

    for h, paths in hash_dict.items():
        if len(paths) > 1:
            folders = set(os.path.dirname(p) for p in paths)

            # Only show if in different folders
            if len(folders) > 1:
                found = True
                print("🔁 Same image found in multiple folders:\n")

                for p in paths:
                    print(f"📂 {p}")

                print("-" * 50)
                total_duplicates += len(paths) - 1

    if found:
        print(f"\n✅ Total duplicate images across folders: {total_duplicates}")
    else:
        print("✅ No same images found across different folders.")

# 🔹 Run
if __name__ == "__main__":
    dataset_path = r"C:\Users\LENOVO\Desktop\image_data\dataset"   # 👈 change this
    check_duplicates(r"C:\Users\LENOVO\Desktop\image_data\dataset")