from collections import Counter
import json
import os
import numpy as np
import math
import random
import torch

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

def print_msg(msg, output_folder="printmsg/", 
              output_file_prefix="msg"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file path
    output_file = os.path.join(output_folder, f"{output_file_prefix}.json")
    
    # Initialize an empty list if the file does not exist or if it is empty
    if not os.path.exists(output_file):
        all_data = []
    else:
        # Read the existing data from the file (if the file exists)
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # Ensure data is in list format
            except json.JSONDecodeError:
                all_data = []  # Handle invalid/empty JSON file
    
    # Append the new message to the data
    all_data.append({"message": msg})
    
    # Write the updated data back to the file at the end of the round
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

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

#average val_f1 from all clients
def average_val_f1(folder):
    val_f1_list = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder, file_name)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    if "val_f1" in data:
                        val_f1_list.append(data["val_f1"])
                except json.JSONDecodeError:
                    print(f"Skipping {file_name} (invalid JSON)")

    if val_f1_list:
        avg_val_f1 = sum(val_f1_list) / len(val_f1_list)
        return avg_val_f1, len(val_f1_list)
    else:
        return None, 0

output_folder = "client_sa_metrics/"

# 1. Check if the client folder exists inside "SA Metrics" and create it if not
def isFirst(client_id, output_folder):
     
    js = os.path.join(output_folder, f"{client_id}.json")
    """Check if the folder for the client exists, and create it if not."""
    # Create the base folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        return True
    
    if not os.path.exists(js):
        return True
    return False

# 2. Read the accuracy from the JSON file for the client
def read_file(client_id, output_folder):
    
    """Read the existing accuracy from the client's JSON file."""
    output_file = os.path.join(output_folder, f"{client_id}.json")
    
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            # print_msg("It is working")
        return existing_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading {output_file}: {e}")
            return None

# 3. Save or update the accuracy in the JSON file for the client
def save_sa(client_id, E, temp, output_folder, update=False):
    
    # update 
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if os.path.exists(output_file):
        existing_data = read_file(client_id, output_folder)
        if update:
            existing_data["E"] = E
            existing_data['temp'] = temp
            
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)

    #new save           
    else:
        client_data = {"E": E, 
                       "temp":temp,
                       }
        with open(output_file, 'w') as f:
            json.dump(client_data, f, indent=4)

# energy calculation for clients
def energy_calc(output_dict):
    acc = output_dict['val_accuracy']
    prec = output_dict['val_precision']
    rec = output_dict['val_recall']
    fpr = output_dict['val_fpr']

    wa = 0.25 # weight for accuracy
    wp = 0.25 # weight for precision
    wr = 0.25 # weight for recall
    wfpr = 0.25 # weight for false positive rate

    E = (wa * acc) + (wp * prec) + (wr * rec) + (wfpr * (1-fpr)) # weights add up to 1
    return E

# SA send model updates or not
def file_handle(client, output_dict, temp):
    global output_folder 
    if type(client) == int or type(client) == str:
        
        if isFirst(client, output_folder): # file not created yet
             if len(output_dict) == 0:
               E = 0
               save_sa(client, E, temp, output_folder) # so make a copy
               return True  # Accept first client regardless
             else:
              E = energy_calc(output_dict)
              save_sa(client, E, temp, output_folder)
              return True  # Accept first client regardless  
        
        else:
            existing_data = read_file(client, output_folder) # read acc
            if existing_data != None:
                prev_E = existing_data.get('E')
                curr_E = energy_calc(output_dict) # get current E

                if curr_E:
                    update = fl_sa(prev_E, curr_E, temp, output_folder, client) # SA below this function
                    if not update:
                        count_update(output_folder, client, 1)
                    save_sa(client, curr_E, temp, output_folder, update=update)
                    return update     # based on sa will update or not
                else:
                    return False  # If curr_E is None/falsy, reject
            else:
                return False  # If existing_data is None, reject
    
    return False  # Default fallback - reject if type check fails

# to keep track of how many times the client did not send the updates by count
def count_update(output_folder, client_id, count):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if os.path.exists(output_file):
        existing_data = read_file(client_id, output_folder)
        if 'count' in existing_data:
            existing_data['count'] += count
        else:
            existing_data['count'] = count

        with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)
        
# simulated annealing function
def fl_sa(prev_E, curr_E, temp, output_folder, client_id):
    if curr_E is None or prev_E is None:
        return False
    
    #  always accept a better solution
    if curr_E > prev_E:
        return True # yes accept weight from the client for aggregation
    
    else:
        # Ensure temperature is positive to avoid division by zero
        if temp <= 0:
            return False

        existing_data = read_file(client_id, output_folder)
        
        if 'count' in existing_data:
            
            r = curr_E/prev_E
            k = 0.01 * math.exp(existing_data['count'])
            temp  = temp * (1 + k * (1 - r))
            
            with open(os.path.join(output_folder, f"{client_id}.json"), 'w') as f:
                json.dump(existing_data, f, indent=4)
            
            

        # change in E and the temperature.
        exp_T = math.exp((curr_E - prev_E) / temp)

        # Generate a random probability between 0 and 1
        random_probability = random.random()

        # The rest of your logic now works correctly
        if exp_T > random_probability:
            return True  # yes accept weight from the client's for aggregation
        
        else:
            return False # no don't take the client's weights
        



# def print_result(path):
#     for i in range(10):
#         dictt = torch.load("E:/Tahmid_SA/Final Working SA/"+path+f"_{i}.pth")
#         print(f'Client {i}: ', dictt.keys())
#         print()    

# for i in ["best_multiclass_model", "best_binary_model", "best_joint_model"]:
#     print_result(i)