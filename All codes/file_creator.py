import pandas as pd
import json
import sqlite3

# Function to create SQLite database from CSV
def create_database_from_csv(csv_file_path, database_path, table_name):
    df = pd.read_csv(csv_file_path)
    conn = sqlite3.connect(database_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

# Function to fetch metadata and convert to JSON
def create_metadata(db_file, json_file, table_name):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch columns for the specified table_name
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()

    metadata = {}

    # Store columns in metadata
    metadata[table_name] = {}
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        metadata[table_name][col_name] = col_type

    # Close connection
    conn.close()

    # Write metadata to JSON file
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata successfully written to {json_file}")

    # Print overview of extracted metadata
    print("\nOverview of Extracted Metadata:")
    print(f"Table: {table_name}")
    for col_name, col_type in metadata[table_name].items():
        print(f"  Column: {col_name}, Type: {col_type}")

def main():
    # Constants (replace with your actual file paths)
    CSV_FILE_PATH = r'C:\Users\sanan\OneDrive\Desktop\Decision point\suppliers.csv'
    DATABASE_PATH = 'Suppliers.db'
    JSON_FILE = 'supplier_metadata.json'
    TABLE_NAME = 'suppliers'  # Replace with the actual sheet name from your CSV

    # Create database from CSV
    create_database_from_csv(CSV_FILE_PATH, DATABASE_PATH, TABLE_NAME)
    print(f"Database created successfully at {DATABASE_PATH}")

    # Create metadata for the specified table_name
    create_metadata(DATABASE_PATH, JSON_FILE, TABLE_NAME)

if __name__ == "__main__":
    main()
