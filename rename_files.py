import os


def rename_files(directory_name):
    # Get a list of all files in the directory
    file_list = os.listdir(directory_name)

    # Sort the files for consistent naming
    file_list.sort()

    # Initialize a counter for numbering the files
    i = 0

    # Iterate through the files and rename them
    for file_name in file_list:
        if file_name.lower().endswith('.jpg'):
            # Generate the new file name
            new_file_name = f"{os.path.basename(directory_name)}_{i}.jpg"

            # Construct the full paths for the old and new files
            old_file_path = os.path.join(directory_name, file_name)
            new_file_path = os.path.join(directory_name, new_file_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)

            # Increment the counter
            i += 1


# Example usage:
directory_name = "images/CDW_whole_fragments/Ceramics"
rename_files(directory_name)
