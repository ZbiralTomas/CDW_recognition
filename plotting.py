from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def test_normal_distribution(dataset):
    variable_name = get_variable_name(dataset)
    # Perform the Shapiro-Wilk test for normality
    shapiro = stats.shapiro(dataset)
    p_value = shapiro[1]

    # Set a significance level (e.g., 0.05)
    alpha = 0.05

    # Check if the p-values are less than the significance level
    if p_value < alpha:
        print(f"{variable_name} are not from a normal distribution")
        return False
    else:
        print(f"{variable_name} are from a normal distribution")
        return True


data = pd.read_csv("csv_files/plot_data/measurement-06_11_23.csv")

ground_truth_weights = data["gt weight[g]"]
measured_weights = data["calculated weight[g]"]
weight_difference = data["weight difference[g]"]
percentage_weight_difference = 100.0 * abs(weight_difference) / ground_truth_weights
if test_normal_distribution(ground_truth_weights) == False or test_normal_distribution(measured_weights) == False:
    weight_correlation_coefficient, _ = stats.spearmanr(ground_truth_weights, measured_weights)
else:
    weight_correlation_coefficient = np.corrcoef(ground_truth_weights, measured_weights)[0, 1]

test_data = [502.69, 466.14, 391.66, 546.1, 578.02, 520.28, 501.92, 501.47, 490.05, 512.81, 590.31, 455.49, 513.1,
             463.59, 419.67, 442.46, 515.31, 481.53, 450.06, 538.11, 455.22, 350.17, 565.94, 550.63, 551.02, 415.12,
             549.73, 446.86, 509.65, 452.25, 581.62, 543.7, 571.22, 570.88, 459.67, 521.22, 516.55, 514.45, 550.79,
             460.01]
test_normal_distribution(test_data)

ground_truth_volumes = data["gt volume[ml]"]
measured_volumes = data["calculated volume[ml]"]
volume_difference = data["volume difference[ml]"]
percentage_volume_difference = 100.0 * abs(volume_difference) / ground_truth_volumes
if test_normal_distribution(ground_truth_volumes) == False or test_normal_distribution(measured_volumes) == False:
    volume_correlation_coefficient, _ = stats.spearmanr(ground_truth_volumes, measured_volumes)
else:
    volume_correlation_coefficient = np.corrcoef(ground_truth_volumes, measured_volumes)[0, 1]

ground_truth_densities = pd.Series([w / v for w, v in zip(ground_truth_weights, ground_truth_volumes)])
measured_densities = pd.Series([w / v for w, v in zip(measured_weights, measured_volumes)])
density_difference = pd.Series([a - b for a, b in zip(measured_densities, ground_truth_densities)])
percentage_density_difference = 100.0 * abs(density_difference) / ground_truth_densities
if test_normal_distribution(ground_truth_densities) == False or test_normal_distribution(measured_densities) == False:
    density_correlation_coefficient, _ = stats.spearmanr(ground_truth_densities, measured_densities)
else:
    density_correlation_coefficient = np.corrcoef(ground_truth_densities, measured_densities)[0, 1]

#plt.rc('text', usetex=True)
fig, axs = plt.subplots(1, 3, figsize=(17, 4))
# Subplot 1: Weight data
axs[0].scatter(ground_truth_weights, measured_weights, c=percentage_weight_difference, cmap='coolwarm', marker='o')
axs[0].plot([min(ground_truth_weights.min(), measured_weights.min()), max(ground_truth_weights.max(),
                                                                          measured_weights.max())],
            [min(ground_truth_weights.min(), measured_weights.min()),
             max(ground_truth_weights.max(), measured_weights.max())], 'k--')
axs[0].set_title(f'{weight_correlation_coefficient:.3f}')
axs[0].set_xlabel('Ground Truth Weight (g)')
axs[0].set_ylabel('Measured Weight (g)')
axs[0].grid(True)
axs[0].set_xlim(0, max(ground_truth_weights.max(), measured_weights.max()) + 50)
axs[0].set_ylim(0, max(ground_truth_weights.max(), measured_weights.max()) + 50)

# Subplot 2: Volume data
axs[1].scatter(ground_truth_volumes, measured_volumes, c=percentage_volume_difference, cmap='coolwarm', marker='o')
axs[1].plot([min(ground_truth_volumes.min(), measured_volumes.min()), max(ground_truth_volumes.max(),
                                                                          measured_volumes.max())],
            [min(ground_truth_volumes.min(), measured_volumes.min()),
             max(ground_truth_volumes.max(), measured_volumes.max())], 'k--')
axs[1].set_title(f'{volume_correlation_coefficient:.3f}')
axs[1].set_xlabel('Ground Truth Volume (cm3)')
axs[1].set_ylabel('Measured Volume (cm3)')
axs[1].grid(True)
axs[1].set_xlim(0, max(ground_truth_volumes.max(), measured_volumes.max()) + 50)
axs[1].set_ylim(0, max(ground_truth_volumes.max(), measured_volumes.max()) + 50)

# Subplot 3: Density data
im = axs[2].scatter(ground_truth_densities, measured_densities, c=percentage_density_difference, cmap='coolwarm',
                    marker='o')
axs[2].plot([min(ground_truth_densities.min(), measured_densities.min()), max(ground_truth_densities.max(),
                                                                              measured_densities.max())],
            [min(ground_truth_densities.min(), measured_densities.min()),
             max(ground_truth_densities.max(), measured_densities.max())], 'k--')
axs[2].set_title(f'{density_correlation_coefficient:.3f}')
axs[2].set_xlabel('Ground Truth Density (g/cm3)')
axs[2].set_ylabel('Measured Density (g/cm3)')
axs[2].grid(True)
axs[2].set_xlim(0, max(ground_truth_densities.max(), measured_densities.max()) + 0.2)
axs[2].set_ylim(0, max(ground_truth_densities.max(), measured_densities.max()) + 0.2)

fig.subplots_adjust(right=0.882, bottom=0.13, wspace=0.28)
cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.80])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('% difference')
plt.show()
fig.savefig('figures/density_algorithm.pdf')
