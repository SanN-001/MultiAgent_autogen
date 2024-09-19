import sqlite3

def create_empty_database(database_path):
    # Connect to SQLite database (will create it if not exists)
    conn = sqlite3.connect(database_path)
    conn.close()
    print(f"Empty database created successfully at {database_path}")

def main():
    # Replace with your desired database file path
    DATABASE_PATH = 'empty_database.db'

    # Create empty database
    create_empty_database(DATABASE_PATH)

if __name__ == "__main__":
    main()
