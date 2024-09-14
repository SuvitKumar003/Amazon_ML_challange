import os
import pandas as pd
import concurrent.futures
import sys

# Update the dataset folder path with forward slashes
DATASET_FOLDER = '../dataset/'

# Print the current working directory
print("Current working directory:", os.getcwd())

# List files in the dataset folder
print("Files in dataset folder:", os.listdir(DATASET_FOLDER))

# Try reading CSV files with error handling
try:
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    print("Successfully read train.csv")
except FileNotFoundError as e:
    print(f"Error: train.csv not found. {e}")

try:
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print("Successfully read test.csv")
except FileNotFoundError as e:
    print(f"Error: test.csv not found. {e}")

# Load the sample_test.csv file
try:
    sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    print("Successfully read sample_test.csv")
except FileNotFoundError as e:
    print(f"Error: sample_test.csv not found. {e}")
    sample_test = pd.DataFrame()  # Empty DataFrame to avoid further errors

# Check if sample_test is not empty and contains 'image_link' column
if not sample_test.empty and 'image_link' in sample_test.columns:
    # Extract image links
    image_links = sample_test['image_link'].tolist()
    print("Image links:", image_links[:5])  # Print first 5 links for verification

    # Function to download a batch of images
    def download_batch(image_links):
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(download_images, image_links)

    # Split image links into batches of 50
    batch_size = 50
    batches = [image_links[i:i+batch_size] for i in range(0, len(image_links), batch_size)]

    # Process each batch
    for batch in batches:
        download_batch(batch)
else:
    print("sample_test.csv is either empty or does not contain 'image_link' column.")