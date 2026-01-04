import kagglehub
import shutil
import os

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("tejasurya/eng-spanish")

print("Dataset downloaded to:", path)

# Define destination directory
destination_dir = "/Users/tayfuncebeci/Desktop/nlpproject2/data"

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Copy files from the download path to our project directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Dataset moved to: {destination_dir}")
