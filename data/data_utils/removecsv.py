import os

def delete_csv_files_in_directory(directory):
    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):  # topdown=False ensures we delete files first, then dirs
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Example usage
delete_csv_files_in_directory(os.path.join("../", "dataset") )
