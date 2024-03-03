import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the directory for saving plots exists, create if it doesn't
plots_directory = "figures/scatter_plots"
os.makedirs(plots_directory, exist_ok=True)

# Load data from CSV
csv_file_path = 'csv_files/new_data.csv'  # Update this to your CSV file path
df = pd.read_csv(csv_file_path)


# Function to create and save plots
def create_and_save_plot(x, y, hue, data, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 8))
    for value in data[hue].unique():
        subset = data[data[hue] == value]
        plt.scatter(subset[x], subset[y], label=value, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # loc and bbox_to_anchor moves legend outside a plot
    plt.legend(title=hue, loc='upper left', bbox_to_anchor=(1, 1))
    # bbox_inches='tight' ensure that the legend is not cut off
    plt.savefig(f'{plots_directory}/{filename}.pdf', bbox_inches='tight')
    plt.close()


# List of features for density comparisons
features = ['mean_r', 'mean_g', 'mean_b', 'brightness', 'relative_r', 'relative_g', 'relative_b', 'entropy', 'mig']

# Generate plots for density vs. each feature, distinguished by material
for feature in features:
    create_and_save_plot('density', feature, 'material', df, 'Density', feature, f'density_vs_{feature}_by_material')

for feature in features:
    create_and_save_plot('mean_height', feature, 'material', df, 'Mean height', feature, f'mean_geight_vs_{feature}_by_material')
# Generate plot for mean_height vs. density, distinguished by material
create_and_save_plot('mean_height', 'density', 'material', df, 'Mean Height', 'Density', 'mean_height_vs_density_by_material')

# Generate plots for mean_height vs. entropy and mig, distinguished by material
create_and_save_plot('mean_height', 'entropy', 'material', df, 'Mean Height', 'Entropy', 'mean_height_vs_entropy_by_material')
create_and_save_plot('mean_height', 'mig', 'material', df, 'Mean Height', 'MIG', 'mean_height_vs_mig_by_material')



