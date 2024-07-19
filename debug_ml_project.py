import os
import pandas as pd

# Define the file path
file_path = r'C:\Users\PC\Downloads\PROJECT AS data.csv'

# Check if the file exists
if os.path.exists(file_path):
    print("File found, attempting to read...")
    try:
        data = pd.read_csv(file_path)
        print("File read successfully!")
        print(data.head())
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File not found. Please check the file path.")
