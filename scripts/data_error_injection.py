# ============================================================
# data_error_injection.py
# ============================================================
#
# SIMPLE EXPLANATION:
# Imagine you have a perfect Excel sheet with taxi data.
# This script deliberately makes some cells wrong.
# Like a teacher adding mistakes to a student's paper
# to see if they can find them!
#
# We add 7 types of mistakes:
# 1. Empty cells (missing values)
# 2. Negative fare (impossible!)
# 3. Too many passengers (impossible!)
# 4. Wrong city coordinates (not New York!)
# 5. Text in a number column (wrong data type!)
# 6. Date in the future (impossible!)
# 7. Dropoff in the ocean (wrong GPS!)
#
# HOW TO RUN:
# python data_error_injection.py
# ============================================================


# ── Import libraries ──────────────────────────────────────────
import pandas as pd    # For reading and editing data tables
import numpy as np     # For math and random number generation
import os              # For creating folders


# ── Settings (change these if you want) ──────────────────────

# Where is your clean data?
INPUT_FILE = "../data/train.csv"

# Where to save the data with errors?
OUTPUT_FILE = "../data/raw/train_with_errors.csv"

# How many rows to use?
NUM_ROWS = 1000

# How many rows should have errors?
# 50 means 50 rows out of 1000 will have each error
NUM_ERROR_ROWS = 50


# ── Step 1: Load the clean data ──────────────────────────────
print("Loading clean data...")
df = pd.read_csv(INPUT_FILE, nrows=NUM_ROWS)
print(f"Loaded {len(df)} rows!")
print(f"Columns: {list(df.columns)}")


# ── Step 2: Make a copy ───────────────────────────────────────
# IMPORTANT: We work on a COPY, not the original!
# Like making a photocopy before writing on it
df_bad = df.copy()

# Convert fare column to allow both numbers and text
# (we need this for Error 5)
df_bad["fare_amount"] = df_bad["fare_amount"].astype(object)

print("\nStarting to add errors...")


# ── ERROR 1: Missing Values ───────────────────────────────────
# Make passenger_count empty in some rows
# Like someone forgot to fill in a form field

# Pick 50 random row numbers
rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)

# Set passenger_count to NaN (empty/missing)
df_bad.loc[rows_to_affect, "passenger_count"] = np.nan

print(f"Error 1 added: {NUM_ERROR_ROWS} rows have missing passenger_count")


# ── ERROR 2: Negative Fare Amount ────────────────────────────
# A taxi fare can NEVER be negative!
# Like getting money back for taking a taxi - impossible!

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "fare_amount"] = -10.50

print(f"Error 2 added: {NUM_ERROR_ROWS} rows have negative fare (-10.50)")


# ── ERROR 3: Too Many Passengers ─────────────────────────────
# A normal taxi fits max 6 passengers
# 100 passengers in one taxi is impossible!

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "passenger_count"] = 100

print(f"Error 3 added: {NUM_ERROR_ROWS} rows have 100 passengers (impossible!)")


# ── ERROR 4: Wrong City GPS Coordinates ──────────────────────
# New York City latitude is around 40 to 42
# New York City longitude is around -74 to -72
# We add coordinates from London (51.5, -0.1) - wrong city!

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "pickup_latitude"] = 51.5    # London latitude!
df_bad.loc[rows_to_affect, "pickup_longitude"] = -0.1   # London longitude!

print(f"Error 4 added: {NUM_ERROR_ROWS} rows have London coordinates (wrong city!)")


# ── ERROR 5: Text in a Number Column ─────────────────────────
# fare_amount should ONLY have numbers like 12.50
# But we put the word "unknown" in some rows
# Like writing "many" instead of "5" in an age column

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "fare_amount"] = "unknown"

print(f"Error 5 added: {NUM_ERROR_ROWS} rows have 'unknown' in fare_amount")


# ── ERROR 6: Date in the Future ───────────────────────────────
# These are historical taxi trips from 2009-2015
# A trip from year 2099 is impossible!

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "pickup_datetime"] = "2099-01-01 00:00:00 UTC"

print(f"Error 6 added: {NUM_ERROR_ROWS} rows have future date (year 2099!)")


# ── ERROR 7: Dropoff in the Ocean ────────────────────────────
# Coordinates 0.0, 0.0 is in the middle of the ocean
# near Africa - definitely not New York!

rows_to_affect = np.random.choice(df_bad.index, size=NUM_ERROR_ROWS, replace=False)
df_bad.loc[rows_to_affect, "dropoff_latitude"] = 0.0
df_bad.loc[rows_to_affect, "dropoff_longitude"] = 0.0

print(f"Error 7 added: {NUM_ERROR_ROWS} rows have dropoff in the ocean (0.0, 0.0)!")


# ── Step 3: Create output folder ───────────────────────────
# exist_ok=True means: don't show error if folder already exists
os.makedirs("../data/raw", exist_ok=True)


# ── Step 4: Save the data with errors ────────────────────────
df_bad.to_csv(OUTPUT_FILE, index=False)
# index=False means: don't save row numbers as a column


# ── Step 5: Print Summary ─────────────────────────────────────
print("\n" + "="*50)
print("SUMMARY OF ERRORS ADDED")
print("="*50)

total_rows = len(df_bad)

print(f"Total rows in file: {total_rows}")
print(f"")
print(f"Error 1 - Missing passenger_count:  {df_bad['passenger_count'].isna().sum()} rows")
print(f"Error 2 - Negative fare:            {(df_bad['fare_amount'] == -10.50).sum()} rows")
print(f"Error 3 - 100 passengers:           {(df_bad['passenger_count'] == 100).sum()} rows")
print(f"Error 4 - London coordinates:       {(df_bad['pickup_latitude'] == 51.5).sum()} rows")
print(f"Error 5 - 'unknown' in fare:        {(df_bad['fare_amount'] == 'unknown').sum()} rows")
print(f"Error 6 - Future date (2099):       {(df_bad['pickup_datetime'] == '2099-01-01 00:00:00 UTC').sum()} rows")
print(f"Error 7 - Ocean dropoff (0,0):      {(df_bad['dropoff_latitude'] == 0.0).sum()} rows")
print(f"")
print(f"File saved to: {OUTPUT_FILE}")
print("="*50)
print("DONE! Your error injection script worked!")