import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math


# Set matplotlib to non-interactive backend to prevent crashes on servers/headless environments
plt.switch_backend('Agg')

def plot_and_save_averaged_metrics(folder, metrics_folder, result_path):
    print(f"\n--- Processing: {folder} ---")
    
    
    metrics_folder = os.path.abspath(metrics_folder)
    result_path = os.path.abspath(result_path)
    source_dir = os.path.join(metrics_folder, folder)
    final_output_dir = os.path.join(result_path, folder)
    os.makedirs(final_output_dir, exist_ok=True)
    json_pattern = os.path.join(source_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"WARNING: No JSON files found matching: {json_pattern}")
        return

    print(f"Found {len(json_files)} client metric files. Aggregating data...")
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
            print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(all_records)

    if df.empty:
        print("WARNING: JSON files existed, but contained no valid data/metrics.")
        return
    try:
        df_avg = df.groupby('round').mean().sort_index()
    except Exception as e:
        print(f"Error during aggregation (Are metrics numeric?): {e}")
        return

    print(f"Aggregated {len(df)} records into {len(df_avg)} rounds.")
    metric_cols = [c for c in df_avg.columns if c != 'round']
    if not metric_cols:
        print("No metric columns found to plot.")
        return

    num_metrics = len(metric_cols)
    cols = 2
    rows = math.ceil(num_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
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
        
        
        if metric.lower() != "loss":
            ax.set_ylim(0, 1)

        

        
        if len(df_avg) < 20:
            for x, y in zip(df_avg.index, df_avg[metric]):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    
    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f"Federated Learning: Aggregated Metrics ({folder})", fontsize=16)

   
    output_image_path = os.path.join(final_output_dir, "aggregated_metrics_plot.png")
    try:
        print(f"Attempting to save plot to: {output_image_path}")
        plt.savefig(output_image_path, dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully.")
    except Exception as e:
        print(f"Failed to save graph: {e}")
        return 

    
    
        
        
        
    




# Set matplotlib to non-interactive backend to prevent crashes on servers/headless environments
plt.switch_backend('Agg')

def plot_and_save_energy_temp(metrics_folder, result_path):
    print(f"\n--- Processing ---")
    
    metrics_folder = os.path.abspath(metrics_folder)
    result_path = os.path.abspath(result_path)
    source_dir = os.path.join(metrics_folder)
    final_output_dir = os.path.join(result_path)
    os.makedirs(final_output_dir, exist_ok=True)
    
    json_pattern = os.path.join(source_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"WARNING: No JSON files found matching: {json_pattern}")
        return

    print(f"Found {len(json_files)} client metric files. Aggregating data...")
    all_records = []

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
                
                for entry in data:
                    # Access E and temp directly
                    record = {
                        'call_number': entry.get('call_number'),
                        'client_id': entry.get('client_id'),
                        'E' : entry.get("E", entry.get("prev_E")),
                        'temp': entry.get('temp')  # Direct access to 'temp'
                    }
                    all_records.append(record)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    df = pd.DataFrame(all_records)
    
    if df.empty:
        print("WARNING: JSON files existed, but contained no valid data/metrics.")
        return
    
    try:
        # Aggregate the data by 'call_number' (rounds)
        df_avg = df.groupby('call_number').mean().sort_index()
    except Exception as e:
        print(f"Error during aggregation (Are metrics numeric?): {e}")
        return
    
    print(f"Aggregated {len(df)} records into {len(df_avg)} rounds.")
    
    # Create the first plot (Average Temperature vs Rounds)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plotting Average Temperature for each round
    ax.plot(df_avg.index, df_avg['temp'], color='g', marker='o', label="Average Temperature")
    ax.set_title("Average Temperature per Round", fontsize=14, fontweight='bold')
    ax.set_xlabel("Round Number", fontsize=12)
    ax.set_ylabel("Average Temperature (temp)", fontsize=12)
    
    # Show grid for clarity
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the first plot
    output_image_path_temp = os.path.join(final_output_dir, "average_temperature_plot.png")
    try:
        print(f"Attempting to save plot to: {output_image_path_temp}")
        plt.savefig(output_image_path_temp, dpi=300)
        plt.close(fig)
        print(f"Temperature graph saved successfully.")
    except Exception as e:
        print(f"Failed to save temperature graph: {e}")
        return
    
    # Create the second plot (Average Energy vs Rounds)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plotting Average Energy for each round
    ax.plot(df_avg.index, df_avg['E'], color='b', marker='o', label="Average Energy")
    ax.set_title("Average Energy per Round", fontsize=14, fontweight='bold')
    ax.set_xlabel("Round Number", fontsize=12)
    ax.set_ylabel("Average Energy (E)", fontsize=12)
    
    # Show grid for clarity
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the second plot
    output_image_path_energy = os.path.join(final_output_dir, "average_energy_plot.png")
    try:
        print(f"Attempting to save plot to: {output_image_path_energy}")
        plt.savefig(output_image_path_energy, dpi=300)
        plt.close(fig)
        print(f"Energy graph saved successfully.")
    except Exception as e:
        print(f"Failed to save energy graph: {e}")
        return
