import os
import json

def save_metrics_to_json(metrics_dict, message, client_id, output_folder="E:/New_IDS - Copy/results/"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare the output dictionary with client id, message, and metrics
    partition_data = {
        "message": message,
        "metrics": metrics_dict
    }
    
    # Define the output file for this client
    output_file = os.path.join(output_folder, f"client_{client_id}_metrics.json")

    # Read existing data from the JSON file, if it exists
    all_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # Convert single dict to list if needed
            except json.JSONDecodeError:
                all_data = []  # Handle empty or invalid JSON file

    # Calculate the call number (number of existing entries + 1)
    call_number = len(all_data) + 1
    partition_data["call_number"] = call_number

    # Append the new data
    all_data.append(partition_data)

    # Write updated data back to the JSON file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

    
