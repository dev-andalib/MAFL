from collections import Counter
import json
import os
import numpy as np

def get_class_distribution(partition_id, dataloader, message, 
                           output_folder="E:/New_IDS - Copy/class_dist", 
                           output_file_prefix="class_distribution"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a Counter to store the frequency of each label
    label_counts = Counter()

    for batch in dataloader:
        # Assume batch is a tuple (features, labels) from TensorDataset
        _, labels = batch  # Unpack the tuple, ignore features
        labels = labels.cpu().numpy()  # Convert to numpy array

        # For binary classification, labels are float tensors of shape [batch, 1]
        # Convert to binary integers (0 or 1) by thresholding
        labels = (labels > 0.5).astype(int).flatten()  # Threshold at 0.5 and flatten

        # Update the Counter with the labels in the current batch
        label_counts.update(labels)

    # Convert int64 keys to Python int for JSON serialization
    label_counts_converted = {int(key): value for key, value in label_counts.items()}

    # Prepare the output dictionary for this call
    partition_data = {
        "Client no": partition_id,
        "message": message,
        "class_distribution": label_counts_converted
    }

    # Define the output file for this partition in the specified output folder
    output_file = os.path.join(output_folder, f"{output_file_prefix}_client_{partition_id}.json")

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