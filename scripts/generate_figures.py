#!/usr/bin/env python3
"""Generate publication-quality figures for the Sempress paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory
output_dir = Path('docs/assets')
output_dir.mkdir(parents=True, exist_ok=True)

# Benchmark data
benchmark_data = {
    'Dataset': ['Telemetry', 'Sensor\nPhysics', 'ML\nFeatures', 'Financial'],
    'Sempress': [8.08, 5.88, 5.46, 3.80],
    'Gzip': [3.58, 2.76, 3.09, 2.51],
    'Bzip2': [5.56, 4.11, 4.50, 3.13],
    'Zstd': [3.35, 2.65, 2.88, 2.40],
    'Numeric_Pct': [60, 87, 87, 80]
}

df = pd.DataFrame(benchmark_data)

# Figure 1: Compression Ratio Comparison (Bar Chart)
print("Generating Figure 1: Compression Ratio Comparison...")
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(df['Dataset']))
width = 0.2

bars1 = ax.bar(x - 1.5*width, df['Sempress'], width, label='Sempress', 
               color='#0ea5e9', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x - 0.5*width, df['Gzip'], width, label='Gzip', 
               color='#94a3b8', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + 0.5*width, df['Bzip2'], width, label='Bzip2', 
               color='#cbd5e1', edgecolor='black', linewidth=0.5)
bars4 = ax.bar(x + 1.5*width, df['Zstd'], width, label='Zstd', 
               color='#e2e8f0', edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Compression Ratio (higher is better)', fontweight='bold')
ax.set_title('Compression Ratios: Sempress vs. Baseline Codecs', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Dataset'])
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 9)

plt.tight_layout()
plt.savefig(output_dir / 'figure1_compression_ratios.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure1_compression_ratios.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure1_compression_ratios.pdf")
plt.close()

# Figure 2: Sempress Improvement Over Gzip
print("\nGenerating Figure 2: Improvement Over Gzip...")
fig, ax = plt.subplots(figsize=(8, 5))

improvement = ((df['Sempress'] - df['Gzip']) / df['Gzip'] * 100).values

colors = ['#0d9488' if x > 70 else '#0ea5e9' for x in improvement]
bars = ax.barh(df['Dataset'], improvement, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvement)):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
            f'+{val:.0f}%', 
            ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Improvement Over Gzip (%)', fontweight='bold')
ax.set_title('Sempress Compression Improvement vs. Gzip', fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'figure2_improvement.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure2_improvement.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure2_improvement.pdf")
plt.close()

# Figure 3: Compression vs. Numeric Density
print("\nGenerating Figure 3: Compression vs. Numeric Density...")
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot with trend lines
ax.scatter(df['Numeric_Pct'], df['Sempress'], s=200, c='#0ea5e9', 
          marker='o', label='Sempress', edgecolors='black', linewidth=1.5, zorder=3)
ax.scatter(df['Numeric_Pct'], df['Gzip'], s=200, c='#94a3b8', 
          marker='s', label='Gzip', edgecolors='black', linewidth=1.5, zorder=3)

# Add trend lines
z_sempress = np.polyfit(df['Numeric_Pct'], df['Sempress'], 1)
z_gzip = np.polyfit(df['Numeric_Pct'], df['Gzip'], 1)
p_sempress = np.poly1d(z_sempress)
p_gzip = np.poly1d(z_gzip)

x_trend = np.linspace(55, 90, 100)
ax.plot(x_trend, p_sempress(x_trend), '--', color='#0ea5e9', linewidth=2, alpha=0.7)
ax.plot(x_trend, p_gzip(x_trend), '--', color='#94a3b8', linewidth=2, alpha=0.7)

# Annotate data points
for i, row in df.iterrows():
    ax.annotate(row['Dataset'], 
               (row['Numeric_Pct'], row['Sempress']), 
               textcoords="offset points", 
               xytext=(0,10), 
               ha='center',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Numeric Column Density (%)', fontweight='bold')
ax.set_ylabel('Compression Ratio', fontweight='bold')
ax.set_title('Compression Ratio vs. Numeric Density', fontweight='bold', pad=20)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(55, 90)
ax.set_ylim(2, 9)

plt.tight_layout()
plt.savefig(output_dir / 'figure3_numeric_density.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure3_numeric_density.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure3_numeric_density.pdf")
plt.close()

# Figure 4: Size Scaling (from existing size_sweep data)
print("\nGenerating Figure 4: Size Scaling Analysis...")
size_sweep_data = {
    'rows': [1000, 5000, 25000, 100000],
    'telemetry_sempress': [4.98, 5.76, 6.92, 8.08],
    'telemetry_gzip': [3.68, 3.70, 3.71, 3.58],
    'retail_sempress': [5.05, 5.69, 6.19, 6.20],
    'retail_gzip': [5.73, 5.91, 5.95, 5.60]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Telemetry scaling
ax1.plot(size_sweep_data['rows'], size_sweep_data['telemetry_sempress'], 
        'o-', linewidth=2.5, markersize=8, color='#0ea5e9', label='Sempress')
ax1.plot(size_sweep_data['rows'], size_sweep_data['telemetry_gzip'], 
        's-', linewidth=2.5, markersize=8, color='#94a3b8', label='Gzip')
ax1.set_xlabel('Dataset Size (rows)', fontweight='bold')
ax1.set_ylabel('Compression Ratio', fontweight='bold')
ax1.set_title('Telemetry Dataset Scaling', fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xscale('log')

# Retail scaling
ax2.plot(size_sweep_data['rows'], size_sweep_data['retail_sempress'], 
        'o-', linewidth=2.5, markersize=8, color='#0ea5e9', label='Sempress')
ax2.plot(size_sweep_data['rows'], size_sweep_data['retail_gzip'], 
        's-', linewidth=2.5, markersize=8, color='#94a3b8', label='Gzip')
ax2.set_xlabel('Dataset Size (rows)', fontweight='bold')
ax2.set_ylabel('Compression Ratio', fontweight='bold')
ax2.set_title('Retail Dataset Scaling', fontweight='bold')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xscale('log')

plt.suptitle('Compression Ratio Scaling with Dataset Size', 
            fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure4_size_scaling.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure4_size_scaling.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure4_size_scaling.pdf")
plt.close()

# Figure 5: Ablation Study - Effect of Codebook Size (k)
print("\nGenerating Figure 5: Codebook Size Ablation...")
k_values = [16, 32, 64, 128, 256]
ratios = [7.12, 7.85, 8.08, 8.15, 8.18]
rmse = [0.18, 0.09, 0.05, 0.02, 0.01]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Compression ratio vs k
ax1.plot(k_values, ratios, 'o-', linewidth=2.5, markersize=10, 
        color='#0ea5e9', markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Codebook Size (k)', fontweight='bold')
ax1.set_ylabel('Compression Ratio', fontweight='bold')
ax1.set_title('Compression Ratio vs. Codebook Size', fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.axvline(x=64, color='red', linestyle='--', alpha=0.5, label='Selected k=64')
ax1.legend(frameon=True)
ax1.set_ylim(6.5, 8.5)

# RMSE vs k
ax2.plot(k_values, rmse, 'o-', linewidth=2.5, markersize=10, 
        color='#f59e0b', markerfacecolor='white', markeredgewidth=2)
ax2.set_xlabel('Codebook Size (k)', fontweight='bold')
ax2.set_ylabel('RMSE (no residuals)', fontweight='bold')
ax2.set_title('Reconstruction Error vs. Codebook Size', fontweight='bold')
ax2.grid(alpha=0.3, linestyle='--')
ax2.axvline(x=64, color='red', linestyle='--', alpha=0.5, label='Selected k=64')
ax2.legend(frameon=True)
ax2.set_yscale('log')

plt.suptitle('Ablation Study: Effect of Codebook Size (Telemetry Dataset)', 
            fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure5_ablation_k.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure5_ablation_k.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure5_ablation_k.pdf")
plt.close()

# Figure 6: Performance Comparison (Encode/Decode Times)
print("\nGenerating Figure 6: Performance Analysis...")
perf_data = {
    'Dataset': ['Telemetry', 'Sensor\nPhysics', 'ML\nFeatures', 'Financial'],
    'Sempress_Decode': [0.28, 0.61, 0.55, 0.28],
    'Gzip_Decode': [0.01, 0.02, 0.02, 0.01],
    'Gzip_Parse': [0.19, 0.48, 0.43, 0.24]
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(perf_data['Dataset']))
width = 0.25

bars1 = ax.bar(x - width, perf_data['Sempress_Decode'], width, 
              label='Sempress Decode', color='#0ea5e9', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, perf_data['Gzip_Decode'], width, 
              label='Gzip Decode Only', color='#cbd5e1', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, perf_data['Gzip_Parse'], width, 
              label='Gzip Decode + CSV Parse', color='#94a3b8', edgecolor='black', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Dataset (100K rows)', fontweight='bold')
ax.set_ylabel('Decode Time (seconds, lower is better)', fontweight='bold')
ax.set_title('Decode Performance: Sempress vs. Gzip+Parse', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(perf_data['Dataset'])
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'figure6_performance.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'figure6_performance.png', bbox_inches='tight')
print(f"  Saved: {output_dir}/figure6_performance.pdf")
plt.close()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
for f in sorted(output_dir.glob('figure*.pdf')):
    print(f"  - {f.name}")
