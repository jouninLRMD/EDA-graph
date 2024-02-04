# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:05:31 2023

@author: lrm22005
"""
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
##################################################### EDA-graph features ####################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler

path_py_file = r'C:\Users\lrm22005\OneDrive - University of Connecticut\Research\emotion_graph\codes\EDA-graph\\'

dir_euclidean = os.path.dirname(path_py_file)

feature_quantized_graph = glob.glob(dir_euclidean + "\*EDA_graph_features.csv")

# Load the graph data and labels
feature_quantized_graph_data = pd.read_csv(feature_quantized_graph[0])

# Generating the labels class
y = feature_quantized_graph_data['class'].reset_index(drop=True)
X_newl = feature_quantized_graph_data.drop(['class','subject','valence','arousal'],axis=1)
X_newl = X_newl.astype(np.float64)

from sklearn.feature_selection import SelectKBest, f_classif

# Create an instance of the UMAP class

# Select the 2 most relevant features
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X_newl, y)

# Get the feature names
features = X_newl.columns
selected_features = features[selector.get_support()]
selected_features = [feature.replace("_", " ") for feature in selected_features]
print('The 5 most relevant features:', selected_features)
# Use the 2 most relevant features for further analysis or modeling
X_relevant_quantized = X_newl[features[selector.get_support()]]

########################################################################################################################################################################################
##################################################### EDA-graph features Normality analysis ############################################################################################
########################################################################################################################################################################################

# Import necessary libraries
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = r'C:\Users\lrm22005\OneDrive - University of Connecticut\Research\emotion_graph\codes\EDA-graph\\'


from scipy.stats import anderson

data = [[X_relevant_quantized[y == cl].iloc[:, i] for cl in np.unique(y)] for i in np.unique(y)]

from scipy.stats import anderson

normality_test_results = []
for i in range(X_relevant_quantized.shape[1]):
    feature_data = [group[i] for group in data]
    for j, group_data in enumerate(feature_data):
        result = anderson(group_data)
        is_normal = result.statistic < result.critical_values[2]  # Using a significance level of 5%
        normality_test_results.append((i, j, result.statistic, result.critical_values, result.significance_level, is_normal))

# Print the results
class_labels = {0: 'N: Neutral', 1: 'A: Amused', 2: 'B: Bored', 3: 'R: Relaxed', 4: 'S: Scared'}
feature_labels = {i+1: feature.replace("_", " ").title() for i, feature in enumerate(selected_features)}

for result in normality_test_results:
    feature_index = result[0] + 1
    class_index = result[1]
    class_label = class_labels[class_index]
    feature_label = feature_labels[feature_index]
    statistic = result[2]
    critical_values = result[3]
    significance_level = result[4]
    is_normal = result[5]
    
    print(f"Class {class_label} Feature {feature_index}: {feature_label}")
    print(f"Anderson-Darling statistic = {statistic}")
    print(f"Critical values = {critical_values}")
    print(f"Significance level = {significance_level}")
    
    if is_normal:
        print("Distribution: Normal")
    else:
        print("Distribution: Non-normal")
    
    print()

import matplotlib.pyplot as plt

p_values = [result[2] for result in normality_test_results]
labels = [f"{class_labels[result[1]]}, {feature_labels[result[0] + 1]}" for result in normality_test_results]
labelss = [f"{class_labels[result[1]].split(': ')[0]}, {feature_labels[result[0] + 1]}" for result in normality_test_results]
colors = ['green' if result[5] else 'red' for result in normality_test_results]

# Plot the p-values
plt.figure(figsize=(10, 6))
bars = plt.bar(labelss, p_values, color=colors)
plt.xticks(labelss, rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Class and Feature", fontsize=25)
plt.ylabel("P-value", fontsize=25)
plt.title("Anderson-Darling Test: P-values for Normality", fontsize=30, fontweight='extra bold')

# Color the bars based on class
class_colors = {'Neutral': 'blue', 'Amused': 'orange', 'Bored': 'green', 'Relaxed': 'purple', 'Scared': 'red'}
for i, bar in enumerate(bars):
    class_label = labels[i].split(',')[0].split(' ')[1]  # Extracting the class label without the "Class " prefix
    bar.set_color(class_colors[class_label])
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, "  Y" if colors[i] == 'green' else "  N", ha='center', va='bottom', rotation=90)

# Add legend
legend_labels = class_labels.values()
legend_colors = class_colors.values()
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors], labels=legend_labels)

plt.tight_layout()
# plt.savefig(path + '\\normality_test.png', dpi=300)
plt.show()

########################################################################################################################################################################################
##################################################### EDA-graph features Significance Analysis ############################################################################################
########################################################################################################################################################################################

from scipy.stats import kruskal
import numpy as np

unique_classes = np.unique(y)

for i in range(X_relevant_quantized.shape[1]):
    feature = X_relevant_quantized.iloc[:, i]
    print(f"Feature {i+1}:")
    
    for j in range(len(unique_classes)):
        for k in range(j, len(unique_classes)):
            class1 = feature[y == unique_classes[j]]
            class2 = feature[y == unique_classes[k]]
            
            statistic, p_value = kruskal(class1, class2)
            print(f"Comparison between class {unique_classes[j]} and class {unique_classes[k]}: p-value = {p_value}")
            
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal
import matplotlib.font_manager as fm

# Define the classes and features
classes = np.unique(y)
num_features = X_relevant_quantized.shape[1]

# Create a figure and axes for the box plot
fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(12, 6), sharey=False)

# Define the class legends
class_legends = {0: 'N: Neutral', 1: 'A: Amused', 2: 'B: Bored', 3: 'R: Relaxed', 4: 'S: Scared'}
class_letters = {0: 'N', 1: 'A', 2: 'B', 3: 'R', 4: 'S'}
class_colors = {0: 'blue', 1: 'green', 2: '#555555', 3: 'purple', 4: 'brown'}  # Updated gray color
features_values = {1: '# shortest', 2: '# count', 3: '# of cliqs', 4: 'distance', 5: 'Units'}

# Iterate over each feature
for i in range(num_features):
    feature_values = [X_relevant_quantized.values[y == c, i] for c in classes]

    # Plot the box plot for the current feature
    boxprops = dict(facecolor='white', linewidth=1.5)
    bp = axes[i].boxplot(feature_values, labels=[class_letters[c] for c in classes], notch=True, whis=(0, 100), bootstrap=10000,  showmeans=True, meanline=True, patch_artist=True, boxprops=boxprops)
    
    # Set colors for each box
    for patch, color in zip(bp['boxes'], [class_colors[c] for c in classes]):
        patch.set_facecolor(color)
    
    # Add mean and standard deviation
    # means = [np.mean(class_values) for class_values in feature_values]
    # stds = [np.std(class_values) for class_values in feature_values]
    # x_pos = np.arange(len(classes)) + 1
    # axes[i].errorbar(x_pos, means, yerr=stds, fmt='o', color='black', markersize=6, capsize=4)
    
    axes[i].set_title(f'{feature_labels[i+1]}', fontsize=25, wrap=True)
    axes[i].set_ylabel(f'{features_values[i+1]}', fontsize=22)
    axes[i].set_xlabel('Class', fontsize=22)

# Create a separate legend for classnames and colors
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=class_colors[c], label=class_legends[c]) for c in classes]
plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(-2.5, 1.3), ncol=len(class_legends))

# Adjust the font size of the tick labels
plt.rc('ytick', labelsize=22)

# Adjust the xticks with class letters
for ax in axes:
    ax.set_xticklabels([class_letters[c] for c in classes], fontsize=22)

# Adjust the spacing between subplots
plt.subplots_adjust(top=0.673, bottom=0.267, left=0.054, right=0.978, hspace=0.2, wspace=0.425)

# Display the plot
# plt.savefig(path + '\\comparison_euclidean_8nn_box_plots_graphs.png', dpi=1000)
plt.show()

import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

# Perform post hoc Dunn test
unique_classes = np.unique(y)

for i in range(X_relevant_quantized.shape[1]):
    feature = X_relevant_quantized.iloc[:, i]
    print(f"Feature {i+1}:")
    
    for j in range(len(unique_classes)):
        for k in range(j+1, len(unique_classes)):
            class1 = feature[y == unique_classes[j]]
            class2 = feature[y == unique_classes[k]]
            
            statistic, p_value = kruskal(class1, class2)
    
    # Perform post hoc Dunn test
    df = pd.DataFrame({'Feature': feature, 'Class': y})
    posthoc_results = sp.posthoc_dunn(df, val_col='Feature', group_col='Class', p_adjust='holm')
    print(posthoc_results)
    # Print the results (significant differences are marked with asterisks)
    significant_results = posthoc_results.applymap(lambda x: '*' if x < 0.005 else '')
    print(significant_results)
    # Get the p-values from the posthoc results
    p_values = posthoc_results.to_numpy().flatten()
    
    # Perform False Discovery Rate (FDR) correction
    reject, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Print the results (significant differences are marked with asterisks after FDR correction)
    corrected_results = np.array(reject).reshape(posthoc_results.shape)
    print(corrected_results)
    
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
##################################################### Traditional EDA features ####################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler

path_py_file_traditional = r'C:\Users\lrm22005\OneDrive - University of Connecticut\Research\emotion_graph\codes\EDA-graph\\'
dir_traditional = os.path.dirname(path_py_file_traditional)

feature_quantized_graph = glob.glob(dir_traditional + "\\EDA_Traditional_Features.csv")

# Load the graph data and labels
feature_quantized_graph_data = pd.read_csv(feature_quantized_graph[0])

# Generating the labels class
y = feature_quantized_graph_data['class'].reset_index(drop=True)

X_newl = feature_quantized_graph_data.drop(['arousal','valence','class'],axis=1)
X_newl = X_newl.astype(np.float64)

from sklearn.feature_selection import SelectKBest, f_classif

# Select the 2 most relevant features
selector = SelectKBest(f_classif, k=4)
X_new = selector.fit_transform(X_newl, y)

# Get the feature names
features = X_newl.columns
selected_features = features[selector.get_support()]
selected_features = [feature.replace("_", " ") for feature in selected_features]
print('The 4 most relevant features:', selected_features)
# Use the 2 most relevant features for further analysis or modeling
X_relevant_quantized = X_newl[features[selector.get_support()]]

########################################################################################################################################################################################
##################################################### EDA Traditional features Normality analysis ############################################################################################
########################################################################################################################################################################################
# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = r'C:\Users\lrm22005\OneDrive - University of Connecticut\Research\emotion_graph\codes\EDA-graph\\'


from scipy.stats import anderson

data = [[X_relevant_quantized[y == cl][feature].values for cl in range(5)] for feature in X_relevant_quantized.columns]

from scipy.stats import anderson

normality_test_results = []

# Iterate over each feature and its corresponding class data
for i, feature_data in enumerate(data):
    for j, group_data in enumerate(feature_data):
        result = anderson(group_data)
        is_normal = result.statistic < result.critical_values[2]  # Using a significance level of 5%
        normality_test_results.append((i, j, result.statistic, result.critical_values, result.significance_level, is_normal))


# Print the results
class_labels = {0: 'N: Neutral', 1: 'A: Amused', 2: 'B: Bored', 3: 'R: Relaxed', 4: 'S: Scared'}
feature_labels = {i+1: feature for i, feature in enumerate(selected_features)}

for result in normality_test_results:
    feature_index = result[0] + 1
    class_index = result[1]
    class_label = class_labels[class_index]
    feature_label = feature_labels[feature_index]
    statistic = result[2]
    critical_values = result[3]
    significance_level = result[4]
    is_normal = result[5]
    
    print(f"Class {class_label} Feature {feature_index}: {feature_label}")
    print(f"Anderson-Darling statistic = {statistic}")
    print(f"Critical values = {critical_values}")
    print(f"Significance level = {significance_level}")
    
    if is_normal:
        print("Distribution: Normal")
    else:
        print("Distribution: Non-normal")
    
    print()

import matplotlib.pyplot as plt

p_values = [result[2] for result in normality_test_results]
labels = [f"{class_labels[result[1]]}, {feature_labels[result[0] + 1]}" for result in normality_test_results]
labelss = [f"{class_labels[result[1]].split(': ')[0]}, {feature_labels[result[0] + 1]}" for result in normality_test_results]
colors = ['green' if result[5] else 'red' for result in normality_test_results]

# Plot the p-values
plt.figure(figsize=(10, 6))
bars = plt.bar(labelss, p_values, color=colors)
plt.xticks(labelss, rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Class and Feature", fontsize=25)
plt.ylabel("P-value", fontsize=25)
plt.title("Anderson-Darling Test: P-values for Normality", fontsize=30, fontweight='extra bold')

# Color the bars based on class
class_colors = {'Neutral': 'blue', 'Amused': 'orange', 'Bored': 'green', 'Relaxed': 'purple', 'Scared': 'red'}
for i, bar in enumerate(bars):
    class_label = labels[i].split(',')[0].split(' ')[1]  # Extracting the class label without the "Class " prefix
    bar.set_color(class_colors[class_label])
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, "  Y" if colors[i] == 'green' else "  N", ha='center', va='bottom', rotation=90)

# Add legend
legend_labels = class_labels.values()
legend_colors = class_colors.values()
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors], labels=legend_labels)

plt.tight_layout()
# plt.savefig(path + '\\normality_test.png', dpi=300)
plt.show()

########################################################################################################################################################################################
##################################################### EDA Traditional features Significance Analysis ############################################################################################
########################################################################################################################################################################################

from scipy.stats import kruskal
import numpy as np

unique_classes = np.unique(y)

for i in range(X_relevant_quantized.shape[1]):
    feature = X_relevant_quantized.iloc[:, i]
    print(f"Feature {i+1}:")
    
    for j in range(len(unique_classes)):
        for k in range(j, len(unique_classes)):
            class1 = feature[y == unique_classes[j]]
            class2 = feature[y == unique_classes[k]]
            
            statistic, p_value = kruskal(class1, class2)
            print(f"Comparison between class {unique_classes[j]} and class {unique_classes[k]}: p-value = {p_value}")
            
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal
import matplotlib.font_manager as fm

# Define the classes and features
classes = np.unique(y)
num_features = X_relevant_quantized.shape[1]

# Create a figure and axes for the box plot
fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(12, 8), sharey=False)  # Increased height to 8

# Define the class legends
class_legends = {0: 'N: Neutral', 1: 'A: Amused', 2: 'B: Bored', 3: 'R: Relaxed', 4: 'S: Scared'}
class_letters = {0: 'N', 1: 'A', 2: 'B', 3: 'R', 4: 'S'}
class_colors = {0: 'blue', 1: 'green', 2: '#555555', 3: 'purple', 4: 'brown'}  # Changed orange to gray
features_values = {1: 'μS', 2: '#/min', 3: 'dimensionless', 4: 'mS$^2$'}
# Define the vertical offsets for differences markers
offsets = [0.15, 0.4, 1, 0.7, 0.5]  # Adjust the offsets as desired

# Iterate over each feature
for i in range(num_features):
    feature_values = [X_relevant_quantized.values[y == c, i] for c in classes]

    # Plot the box plot for the current feature
    boxprops = dict(facecolor='white', linewidth=2)
    bp = axes[i].boxplot(feature_values, labels=[class_letters[c] for c in classes], notch=True, whis=(0, 100), bootstrap=10000,  showmeans=True, meanline=True, patch_artist=True, boxprops=boxprops)
    
    # Set colors for each box
    for patch, color in zip(bp['boxes'], [class_colors[c] for c in classes]):
        patch.set_facecolor(color)
    # Add mean and standard deviation
    means = [np.mean(class_values) for class_values in feature_values]
    stds = [np.std(class_values) for class_values in feature_values]
    x_pos = np.arange(len(classes)) + 1
    axes[i].errorbar(x_pos, means, yerr=stds, fmt='o', color='black', markersize=6, capsize=4)
    
    axes[i].set_title(f'{feature_labels[i+1]}', fontsize=25, wrap=True)
    axes[i].set_ylabel(f'{features_values[i+1]}', fontsize=24)
    axes[i].set_xlabel('Class', fontsize=24)
    
# Create a separate legend for class names and colors
legend_handles = [plt.Rectangle(
    (0, 0), 1, 1, color=class_colors[c], label=class_legends[c]) for c in classes]
plt.legend(handles=legend_handles, loc='upper center', ncol=len(classes), bbox_to_anchor=(-1.8, 1.4))

# Adjust the font size of the tick labels
plt.rc('ytick', labelsize=18)

# Adjust the x-axis tick labels
for ax in axes:
    ax.set_xticklabels([class_letters[c] for c in classes], fontsize=18)

# Adjust the spacing between subplots
plt.subplots_adjust(top=0.658, bottom=0.257, left=0.059, right=0.990, hspace=0.2, wspace=0.455)

# Display the plot
# plt.savefig(path + '\\comparison_box_plots_traditionals_lab_features_reviwed.png', dpi=1000)
plt.show()

import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

# Perform post hoc Dunn test
unique_classes = np.unique(y)

for i in range(X_relevant_quantized.shape[1]):
    feature = X_relevant_quantized.iloc[:, i]
    print(f"Feature {i+1}:")
    
    for j in range(len(unique_classes)):
        for k in range(j+1, len(unique_classes)):
            class1 = feature[y == unique_classes[j]]
            class2 = feature[y == unique_classes[k]]
            
            statistic, p_value = kruskal(class1, class2)
            # print(f"Comparison between class {unique_classes[j]} and class {unique_classes[k]}: p-value = {p_value}")
    
    # Perform post hoc Dunn test
    df = pd.DataFrame({'Feature': feature, 'Class': y})
    posthoc_results = sp.posthoc_dunn(df, val_col='Feature', group_col='Class', p_adjust='holm')
    print(posthoc_results)
    # Print the results (significant differences are marked with asterisks)
    significant_results = posthoc_results.applymap(lambda x: '*' if x < 0.005 else '')
    print(significant_results)
    
    # Get the p-values from the posthoc results
    p_values = posthoc_results.to_numpy().flatten()
    
    # Perform False Discovery Rate (FDR) correction
    reject, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Print the results (significant differences are marked with asterisks after FDR correction)
    corrected_results = np.array(reject).reshape(posthoc_results.shape)
    print(corrected_results)