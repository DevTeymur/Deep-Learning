import os

folder_path = "results/"  # ← change to your actual folder

for filename in os.listdir(folder_path):
    if ":" in filename:
        new_name = filename.replace(":", "_")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
