from collections import Counter
import json
import os
import numpy as np
import math
import random
import torch
import shutil

##################################################
#           BASIC UTILITY FUNCTION               #
##################################################
def save_metrics_graphs(metrics_dict, client_id, file_name, output_folder=r"D:\T24\MAFL\ANDALIB_SA\client_metrics"):
    
    # 1. Define the directory path
    target_directory = os.path.join(output_folder, file_name)
    
    # 2. Create the DIRECTORY
    os.makedirs(target_directory, exist_ok=True)

    # 3. Define the full FILE path
    output_file_path = os.path.join(target_directory, f"client_{client_id}_metrics.json")

    # [SAFETY FIX] Check if a FOLDER exists with the same name as the file
    if os.path.exists(output_file_path) and os.path.isdir(output_file_path):
        try:
            shutil.rmtree(output_file_path)  # Force delete the folder
        except OSError as e:
            pass

    # 4. Initialize list to hold data
    existing_data = []

    # 5. Read existing data
    if os.path.exists(output_file_path) and os.path.isfile(output_file_path):
        try:
            if os.stat(output_file_path).st_size > 0:
                with open(output_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        existing_data = loaded_data
                    elif isinstance(loaded_data, dict):
                        existing_data = [loaded_data]
        except (json.JSONDecodeError, OSError) as e:
            existing_data = []  # If corrupted, start over

    # 6. Calculate averages for any list-type metrics
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            metrics_dict[key] = sum(value) / len(value) if value else 0

    # 7. Prepare the new data entry
    call_number = len(existing_data) + 1
    new_entry = {
        "call_number": call_number,
        "client_id": client_id,
        "metrics": metrics_dict
    }

    # 8. Append and Write the new entry to the existing data
    existing_data.append(new_entry)

    with open(output_file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

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
    
    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

##################################################
#                       SA                       #
##################################################
def isFirst(client_id, output_folder):
    js = os.path.join(output_folder, f"{client_id}.json")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        return True
    
    if not os.path.exists(js):
        return True
    return False

def read_file(client_id, output_folder):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    with open(output_file, 'r') as f:
        return json.load(f)

def save_sa(client_id, E, temp, output_folder):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    
    if os.path.exists(output_file):
        existing_data = read_file(client_id, output_folder)
        call_number = len(existing_data) + 1
        new_entry = {
            "call_number": call_number,
            "client_id": client_id,
            "E": E,
            "temp": temp
        }
        if len(existing_data) > 0:
            if 'count' in existing_data[-1]:
                new_entry['count'] = existing_data[-1]['count']
                
        existing_data.append(new_entry)
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        client_data = [{
            "call_number": 1,
            "client_id": client_id,
            "E": E,
            "temp": temp
        }]
        with open(output_file, 'w') as f:
            json.dump(client_data, f, indent=4)

def energy_calc(output_dict): 
    acc = output_dict['val_accuracy']
    f1 = output_dict['val_f1']
    fpr = output_dict['val_fpr']

    wa = 0.25
    wr = 0.25
    wfpr = 0.5

    E = (wa * acc) + (wr * f1) + (1 - fpr) * wfpr
    return E

def file_handle(client, output_dict, temp, output_folder=r"client_sa_metrics/"):
    if isinstance(client, (int, str)):
        if isFirst(client, output_folder): 
            if len(output_dict) == 0:
                E = 0
                save_sa(client, E, temp, output_folder)
                return True
            else:
                E = energy_calc(output_dict)
                save_sa(client, E, temp, output_folder)
                return True
        
        else:
            existing_data = read_file(client, output_folder)  
            if existing_data:
                latest_entry = existing_data[-1]  
                prev_E = latest_entry.get('E', 0)  
                curr_E = energy_calc(output_dict)  

                if curr_E:
                    update = fl_sa(prev_E, curr_E, temp, output_folder, client)  
                    save_sa(client, curr_E, temp, output_folder)  
                    if not update:
                        count_update(output_folder, client, 1)  
                    
                    return update     
            else:
                return False  
    return False  

def count_update(output_folder, client_id, count):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    
    if os.path.exists(output_file):
        existing_data = read_file(client_id, output_folder)
        latest_entry = existing_data[-1] if existing_data else None
        
        if latest_entry:
            if 'count' in latest_entry:
                latest_entry['count'] += count
            else:
                latest_entry['count'] = count
        
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

def fl_sa(prev_E, curr_E, temp, output_folder, client_id):
    if curr_E is None or prev_E is None:
        return False
    
    if curr_E > prev_E:
        return True 
    
    if temp <= 0:
        return False

    existing_data = read_file(client_id, output_folder)
    
    if 'count' in existing_data[-1]:
        r = curr_E / prev_E
        k = 0.01 * math.exp(existing_data[-1]['count'])
        temp = temp * (1 + k * (1 - r))

        with open(os.path.join(output_folder, f"{client_id}.json"), 'w') as f:
            json.dump(existing_data, f, indent=4)

    exp_T = math.exp((curr_E - prev_E) / temp)
    random_probability = random.random()

    if exp_T > random_probability:
        return True  
    else:
        return False  
