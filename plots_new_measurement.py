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
def create_and_save_plot(x, y, hue, legend, data, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 8))
    for value in data[hue].unique():
        subset = data[data[hue] == value]
        plt.scatter(subset[x], subset[y], label=value, alpha=0.7)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # loc and bbox_to_anchor moves legend outside a plot
    plt.legend(title="materiál", labels=legend, loc='upper left', bbox_to_anchor=(1, 1))
    # bbox_inches='tight' ensure that the legend is not cut off
    plt.savefig(f'{plots_directory}/{filename}.pdf', bbox_inches='tight')
    plt.close()


# List of features for density comparisons
materialy = ['asfalt', 'porobeton', 'polystyren', 'malta', 'keramická cihla', 'sádrovláknitá deska', 'beton', 'sádrokarton', 'glazurovaná strana dlaždice', 'lepená strana dlaždice', 'dřevo']
features = ['mean_r', 'mean_g', 'mean_b', 'brightness', 'relative_r', 'relative_g', 'relative_b', 'entropy', 'mig']
legend = ['průměrná intenzita červené barvy', 'průměrná intenzita zelené barvy', 'průměrná intenzita modré barvy', 'celková intenzita barvy', 'relativní intenzita červené barvy', 'relativní intenzita zelené barvy', 'relativní intenzita modré barvy', 'Shannonova entropie', 'průměrný gradient intenzity']

# Generate plots for density vs. each feature, distinguished by material
for feature, ylabel in zip(features, legend):
    create_and_save_plot('density', feature, 'material', materialy, df, 'hustota (kg/m3)', ylabel, f'density_vs_{feature}_by_material')

for feature in features:
    create_and_save_plot('mean_height', feature, 'material', legend, df, 'Mean height', feature, f'mean_geight_vs_{feature}_by_material')
# Generate plot for mean_height vs. density, distinguished by material
create_and_save_plot('mean_height', 'density', 'material', legend, df, 'Mean Height', 'Density', 'mean_height_vs_density_by_material')

# Generate plots for mean_height vs. entropy and mig, distinguished by material
create_and_save_plot('mean_height', 'entropy', 'material', legend, df, 'Mean Height', 'Entropy', 'mean_height_vs_entropy_by_material')
create_and_save_plot('mean_height', 'mig', 'material', legend, df, 'Mean Height', 'MIG', 'mean_height_vs_mig_by_material')



