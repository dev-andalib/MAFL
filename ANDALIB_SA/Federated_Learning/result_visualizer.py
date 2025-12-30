import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
import shutil

# Set matplotlib to non-interactive backend to prevent crashes on servers/headless environments
plt.switch_backend('Agg')

def plot_and_save_averaged_metrics(folder, metrics_folder, result_path):
    print(f"\n--- Processing: {folder} ---")
    
    # 1. Setup Absolute Paths (Crucial for Windows)
    # Ensure we are working with absolute paths to avoid "File not found" errors
    metrics_folder = os.path.abspath(metrics_folder)
    result_path = os.path.abspath(result_path)

    # Define Source (Where JSONs are) and Destination (Where Image goes)
    source_dir = os.path.join(metrics_folder, folder)
    final_output_dir = os.path.join(result_path, folder)

    print(f"Looking for metrics in: {source_dir}")
    print(f"Saving results to:      {final_output_dir}")

    # 2. Create Output Directory
    os.makedirs(final_output_dir, exist_ok=True)

    # 3. Find JSON Files
    json_pattern = os.path.join(source_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"❌ WARNING: No JSON files found matching: {json_pattern}")
        # If no files, we return early. This explains why no image is saved.
        return

    print(f"✅ Found {len(json_files)} client metric files. Aggregating data...")

    # 4. Load and Aggregate Data
    all_records = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
                
                for entry in data:
                    record = entry.get('metrics', {}).copy()
                    record['round'] = entry.get('call_number')
                    all_records.append(record)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    df = pd.DataFrame(all_records)

    if df.empty:
        print("❌ WARNING: JSON files existed, but contained no valid data/metrics.")
        return

    # Calculate Averages
    try:
        df_avg = df.groupby('round').mean().sort_index()
    except Exception as e:
        print(f"❌ Error during aggregation (Are metrics numeric?): {e}")
        return

    print(f"Aggregated {len(df)} records into {len(df_avg)} rounds.")

    # 5. Plotting
    metric_cols = [c for c in df_avg.columns if c != 'round']
    
    if not metric_cols:
        print("❌ No metric columns found to plot.")
        return

    num_metrics = len(metric_cols)
    cols = 2
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Flatten axes for easy iteration, handle single-subplot case
    if num_metrics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        ax.plot(df_avg.index, df_avg[metric], marker='o', linestyle='-', linewidth=2, label='Average')
        ax.set_title(f"Avg {metric.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Round Number")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate points
        if len(df_avg) < 20:
            for x, y in zip(df_avg.index, df_avg[metric]):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f"Federated Learning: Aggregated Metrics ({folder})", fontsize=16)

    # 6. Save Graph
    # FIX: Use 'final_output_dir' variable we defined at the top
    output_image_path = os.path.join(final_output_dir, "aggregated_metrics_plot.png")
    
    try:
        print(f"Attempting to save plot to: {output_image_path}")
        plt.savefig(output_image_path, dpi=300)
        plt.close(fig)
        print(f"✅ Graph saved successfully.")
    except Exception as e:
        print(f"❌ Failed to save graph: {e}")
        return # Do not delete files if save failed

    # 7. Delete Source Folder
    try:
        print(f"Cleaning up source folder: {source_dir}")
        shutil.rmtree(source_dir)
        print("✅ Cleanup complete.")
    except Exception as e:
        print(f"⚠️ Error deleting source folder: {e}")

