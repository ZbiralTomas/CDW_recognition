import pickle
import matplotlib.pyplot as plt
import os


def get_filenames_with_full_path(directory_path):
    try:
        # List to hold file paths
        filepaths = []

        # os.listdir() returns a list of all files and directories in the specified path
        for entry in os.listdir(directory_path):
            # Construct a full path to the entry
            full_path = os.path.join(directory_path, entry)

            # Check if the entry is a file and not a directory
            if os.path.isfile(full_path):
                filepaths.append(full_path)  # Add the full path

        return filepaths
    except Exception as e:
        # In case of any error, return the error message
        return str(e)


directory_with_plots = 'figures/surfplots/asphalt_cylinders_17'
paths = get_filenames_with_full_path(directory_with_plots)


for path in paths:
    with open(path, 'rb') as f:
        fig = pickle.load(f)

    # Create a new pyplot figure
    new_fig = plt.figure()
    # Get the current figure's manager and set the canvas to the loaded figure's canvas
    new_manager = new_fig.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    # Now you can display the figure with pyplot's functionality
    plt.show()
