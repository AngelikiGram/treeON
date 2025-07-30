import os

# Your source and output folders
folder = "F:\TREES\deciduous" # F:/TREES/deciduous" # conifers"
out_folder = "F:\TREES\DATASET\TREES-temp" # deciduous_renamed" # conifers_renamed"
os.makedirs(out_folder, exist_ok=True)


for fname in os.listdir(folder):
    if fname.startswith("tree_") and fname.endswith(".obj"):
        parts = fname.split("_")
        if len(parts) != 2:
            continue

        number_part = parts[1].split(".")[0]
        try:
            number = int(number_part)
            new_fname = f"tree_{number:04d}.obj"
            old_path = os.path.join(folder, fname)
            new_path = os.path.join(out_folder, new_fname)
            os.rename(old_path, new_path)
            print(f"Renamed {fname} → {new_fname}")
        except ValueError:
            print(f"Skipping invalid filename: {fname}")

# # Collect and sort all relevant .obj files
# obj_files = sorted([f for f in os.listdir(folder) if f.endswith(".obj")]) #  and f.startswith("tree_")])

# for i, fname in enumerate(obj_files, start=1):
#     num = i + 1000 # for deciduous 
#     new_name = f"tree_{num:03d}.obj"
#     old_path = os.path.join(folder, fname)
#     new_path = os.path.join(out_folder, new_name)
#     try:
#         os.rename(old_path, new_path)
#         print(f"Renamed {fname} -> {new_name}")
#     except Exception as e:
#         print(f"❌ Failed to rename {fname}: {e}")

# python 2_rename.py

# then delete the old folder
# and rename the new folder to the old one