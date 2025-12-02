import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

# Load data
data_path = r"C:\Users\BRAIS\Desktop\FPADISTANCIA\BD\analise-datos-pandas-main\analise-datos-pandas-main\data\penguins.csv"
print(f"Loading data from: {data_path}")
df_penguins = pd.read_csv(data_path)

# Create output directory for plots
output_dir = "solutions_plots"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to: {output_dir}")

# --- Exercise 1 ---
print("Solving Exercise 1...")
# Line plot: bill_length_mm vs bill_depth_mm for 'Adelie', sorted by bill_length_mm
df_adelie = df_penguins[df_penguins['species'] == 'Adelie'].sort_values('bill_length_mm')
plt.figure(figsize=(10, 6))
plt.plot(df_adelie['bill_length_mm'], df_adelie['bill_depth_mm'])
plt.title('Relación Longitud vs Profundidad (Adelie)')
plt.xlabel('Lonxitude do peteiro (mm)')
plt.ylabel('Profundidade do peteiro (mm)')
plt.savefig(f"{output_dir}/ex1.png")
plt.close()

# --- Exercise 2 ---
print("Solving Exercise 2...")
# Scatter plot: bill_length_mm vs bill_depth_mm
plt.figure(figsize=(10, 6))
plt.scatter(df_penguins['bill_length_mm'], df_penguins['bill_depth_mm'])
plt.title('Relación Longitud vs Profundidad (Todos)')
plt.xlabel('Lonxitude do peteiro (mm)')
plt.ylabel('Profundidade do peteiro (mm)')
plt.savefig(f"{output_dir}/ex2.png")
plt.close()

# --- Exercise 3 ---
print("Solving Exercise 3...")
# Scatter plot (Seaborn): hue='species'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_penguins, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title('Relación Longitud vs Profundidad por Especie')
plt.savefig(f"{output_dir}/ex3.png")
plt.close()

# --- Exercise 4 ---
print("Solving Exercise 4...")
# Bar plot: Mean body_mass_g per species
mean_mass = df_penguins.groupby('species')['body_mass_g'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.bar(mean_mass['species'], mean_mass['body_mass_g'])
plt.title('Peso medio por especie')
plt.xlabel('Especie')
plt.ylabel('Peso medio (g)')
plt.savefig(f"{output_dir}/ex4.png")
plt.close()

# --- Exercise 5 ---
print("Solving Exercise 5...")
# Two histograms: flipper_length_mm (all) and flipper_length_mm (by species)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram 1: All
axes[0].hist(df_penguins['flipper_length_mm'].dropna(), bins=20, edgecolor='black')
axes[0].set_title('Histograma Lonxitude das ás (Todos)')
axes[0].set_xlabel('Lonxitude das ás (mm)')

# Histogram 2: By species
for species in df_penguins['species'].unique():
    subset = df_penguins[df_penguins['species'] == species]
    axes[1].hist(subset['flipper_length_mm'].dropna(), alpha=0.5, label=species, bins=20, edgecolor='black')
axes[1].set_title('Histograma Lonxitude das ás por Especie')
axes[1].set_xlabel('Lonxitude das ás (mm)')
axes[1].legend()

plt.savefig(f"{output_dir}/ex5.png")
plt.close()

# --- Exercise 6 ---
print("Solving Exercise 6...")
# Boxplot and Violinplot: body_mass_g by sex within species
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df_penguins, x='species', y='body_mass_g', hue='sex', ax=axes[0])
axes[0].set_title('Boxplot Peso por Sexo e Especie')

sns.violinplot(data=df_penguins, x='species', y='body_mass_g', hue='sex', split=True, ax=axes[1])
axes[1].set_title('Violinplot Peso por Sexo e Especie')

plt.savefig(f"{output_dir}/ex6.png")
plt.close()

# --- Exercise 7 ---
print("Solving Exercise 7...")
# Pairplot: numeric variables, hue='species', diag_kind='hist', kind='kde'
# Note: Pairplot creates its own figure
g = sns.pairplot(df_penguins, hue='species', diag_kind='hist', kind='kde')
g.fig.suptitle('Pairplot de variables numéricas', y=1.02)
plt.savefig(f"{output_dir}/ex7.png")
plt.close()

# --- Exercise 8 ---
print("Solving Exercise 8...")
# ECDF (Matplotlib only): bill_length_mm
def ecdf(data):
    x = np.sort(data)
    n = len(data)
    y = np.arange(1, n+1) / n
    return x, y

x, y = ecdf(df_penguins['bill_length_mm'].dropna())
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='.', linestyle='none')
plt.title('ECDF Lonxitude do peteiro (Matplotlib)')
plt.xlabel('Lonxitude do peteiro (mm)')
plt.ylabel('ECDF')
plt.savefig(f"{output_dir}/ex8.png")
plt.close()

# --- Exercise 9 ---
print("Solving Exercise 9...")
# ECDF (Seaborn): bill_length_mm
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df_penguins, x='bill_length_mm')
plt.title('ECDF Lonxitude do peteiro (Seaborn)')
plt.savefig(f"{output_dir}/ex9.png")
plt.close()

# --- Exercise 10 ---
print("Solving Exercise 10...")
# Grid (3 plots): Hist body_mass_g (bins=20), Kde body_mass_g, Ecdf body_mass_g
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Histogram
sns.histplot(data=df_penguins, x='body_mass_g', bins=20, ax=axes[0])
axes[0].set_title('Histograma Peso')

# 2. KDE
sns.kdeplot(data=df_penguins, x='body_mass_g', ax=axes[1], fill=True)
axes[1].set_title('KDE Peso')

# 3. ECDF
sns.ecdfplot(data=df_penguins, x='body_mass_g', ax=axes[2])
axes[2].set_title('ECDF Peso')

plt.tight_layout()
plt.savefig(f"{output_dir}/ex10.png")
plt.close()

# --- Exercise 11 ---
print("Solving Exercise 11...")
# Distribution (Kde): body_mass_g, col='island', hue='species'
g = sns.displot(data=df_penguins, x='body_mass_g', hue='species', col='island', kind='kde', fill=True)
g.fig.suptitle('Distribución do Peso por Illa e Especie', y=1.02)
plt.savefig(f"{output_dir}/ex11.png")
plt.close()

# --- Exercise 12 ---
print("Solving Exercise 12...")
# Heatmap: Correlations of numeric variables
plt.figure(figsize=(10, 8))
numeric_df = df_penguins.select_dtypes(include='number')
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de calor de correlacións')
plt.savefig(f"{output_dir}/ex12.png")
plt.close()

print("All exercises completed successfully!")
