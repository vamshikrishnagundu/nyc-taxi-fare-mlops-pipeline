# ============================================================
# split_dataset.py
# ============================================================
# WHAT THIS FILE DOES IN SIMPLE WORDS:
# Imagine you have one BIG notebook with 10000 pages
# This script tears it into 50 SMALL notebooks
# Each small notebook has 200 pages
# These small notebooks go into a folder called "raw-data"
# ============================================================


# ── STEP 1: Import libraries we need ──
# pandas helps us read and work with CSV files (like Excel in Python)
import pandas as pd

# os helps us create folders and work with files on computer
import os

# numpy helps us do math and split data
import numpy as np


# ── STEP 2: Set up our settings ──
# This is where your big CSV file is located
DATASET_PATH = "../data/train.csv"

# This is the name of the folder where we save small files
RAW_DATA_FOLDER = "../data/raw"

# This is how many small files we want to create
NUM_FILES = 100


# ── STEP 3: Create the raw-data folder ──
# os.makedirs creates the folder on your computer
# exist_ok=True means: if folder already exists, don't show error
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

# Tell the user what happened
print("Step 1 done: raw-data folder created!")


# ── STEP 4: Load the big CSV file ──
# pd.read_csv opens the CSV file and loads it into memory
# nrows=10000 means: only read first 10000 rows (file is too big otherwise)
print("Step 2: Loading the dataset... please wait...")
df = pd.read_csv(DATASET_PATH, nrows=10000)

# Tell the user how many rows were loaded
print(f"Step 2 done: Loaded {len(df)} rows from train.csv!")


# ── STEP 5: Split the data into equal parts ──
# np.array_split splits the dataframe into NUM_FILES equal pieces
# Example: 10000 rows split into 50 pieces = 200 rows each piece
print(f"Step 3: Splitting data into {NUM_FILES} files...")
file_chunks = np.array_split(df, NUM_FILES)

# file_chunks is now a LIST of 50 small dataframes
# file_chunks[0] = first 200 rows
# file_chunks[1] = next 200 rows
# ... and so on


# ── STEP 6: Save each piece as a separate CSV file ──
# enumerate() gives us both the INDEX (i) and VALUE (chunk) of each piece
# i = 0, 1, 2, 3 ... 49
# chunk = the actual data (200 rows)

for i, chunk in enumerate(file_chunks):

    # Create a filename for each file
    # zfill(3) adds zeros in front: 1 becomes 001, 2 becomes 002
    # This keeps files in correct order when sorted
    # Result: raw_data_001.csv, raw_data_002.csv ... raw_data_050.csv
    filename = f"raw_data_{str(i+1).zfill(3)}.csv"

    # Create the full path: raw-data/raw_data_001.csv
    # os.path.join combines folder name and filename correctly
    filepath = os.path.join(RAW_DATA_FOLDER, filename)

    # Save the chunk as a CSV file
    # index=False means: don't save the row numbers (0,1,2...) as a column
    chunk.to_csv(filepath, index=False)

    # Tell user which file was saved
    print(f"Saved file {i+1} of {NUM_FILES}: {filename} ({len(chunk)} rows)")


# ── STEP 7: Final message ──
print("")
print("====================================")
print("ALL DONE!")
print(f"Created {NUM_FILES} files in '{RAW_DATA_FOLDER}' folder")
print(f"Each file has about {len(df) // NUM_FILES} rows")
print("====================================")