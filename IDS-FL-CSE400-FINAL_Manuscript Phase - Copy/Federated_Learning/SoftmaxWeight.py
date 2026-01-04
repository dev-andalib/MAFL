from collections import Counter
import json
import os
import numpy as np
import math
import random
import shutil

##################################################
#           BASIC UTILITY FUNCTION               #
##################################################
def save_metrics_graphs(metrics_dict, client_id, file_name, output_folder=r"client_metrics/"):
    
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

output_folder = "client_sa_metrics/"


# 1. Check if the client file exists, create base folder if not
def isFirst(client_id, output_folder):
    js = os.path.join(output_folder, f"{client_id}.json")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        return True
    if not os.path.exists(js):
        return True
    return False


# 2. Read client history (ALWAYS returns a LIST of entries)
def read_file(client_id, output_folder):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if not os.path.exists(output_file):
        return []

    with open(output_file, "r") as f:
        data = json.load(f)

    # Backward compatibility: if old format was a dict, wrap it into a list
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


# 3. Save SA metrics iteratively (append one entry per call)
def save_sa(client_id, output_dict, temp, output_folder, weights, update=False):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{client_id}.json")

    history = read_file(client_id, output_folder)

    # Energy calculation (same logic as yours)
    curr_metrics = [
        output_dict.get("val_accuracy"),
        output_dict.get("val_f1"),
        1 - output_dict.get("val_fpr"),
    ]
    curr_E = weights.compute_energy(curr_metrics)

    # Build new entry (one per round)
    new_entry = {
        "call_number": len(history) + 1,
        "client_id": client_id,
        "val_accuracy": output_dict.get("val_accuracy"),
        "val_f1": output_dict.get("val_f1"),
        "val_fpr": output_dict.get("val_fpr"),
        "temp": temp,
        "prev_E": curr_E,
        "theta": weights.theta.tolist(),
        "update": bool(update),
    }

    # Carry count forward if it exists in the previous entry
    if len(history) > 0 and "count" in history[-1]:
        new_entry["count"] = history[-1]["count"]

    history.append(new_entry)

    with open(output_file, "w") as f:
        json.dump(history, f, indent=4)


# to keep track of how many times the client did not send the updates by count
def count_update(output_folder, client_id, count):
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if not os.path.exists(output_file):
        return

    history = read_file(client_id, output_folder)
    if not history:
        return

    # ✅ Only modify the LAST entry
    last_entry = history[-1]

    # ✅ Update count on LAST entry only (create if missing on that last entry)
    if "count" in last_entry:
        last_entry["count"] += count
    else:
        last_entry["count"] = count

    with open(output_file, "w") as f:
        json.dump(history, f, indent=4)


# simulated annealing function
def fl_sa(prev_E, curr_E, temp, output_folder, client_id):
    if curr_E is None or prev_E is None:
        return False

    # always accept a better solution
    if curr_E > prev_E:
        return True

    # Ensure temperature is positive to avoid division by zero
    if temp <= 0:
        return False

    history = read_file(client_id, output_folder)
    last_entry = history[-1] if history else {}

    # your same logic, but "count" lives in LAST entry now
    if "count" in last_entry:
        r = curr_E / prev_E
        k = 0.01 * math.exp(last_entry["count"])
        temp = temp * (1 + k * (1 - r))

        # store updated temp back into LAST entry only (same file, iterative history)
        last_entry["temp"] = temp
        with open(os.path.join(output_folder, f"{client_id}.json"), "w") as f:
            json.dump(history, f, indent=4)

    exp_T = math.exp((curr_E - prev_E) / temp)
    random_probability = random.random()

    if exp_T > random_probability:
        return True
    else:
        return False


# SA send model updates or not
def file_handle(client, output_dict, temp):
    global output_folder

    if type(client) == int or type(client) == str:

        # First time: accept
        if isFirst(client, output_folder):
            weights = SoftmaxWeightUpdater()
            save_sa(client, output_dict, temp, output_folder, weights, update=True)
            return True

        # Not first time
        history = read_file(client, output_folder)
        if history is None or len(history) == 0:
            return False

        last_entry = history[-1]
        prev_E = last_entry.get("prev_E")
        prev_theta = last_entry.get("theta")

        curr_metrics = [
            output_dict.get("val_accuracy"),
            output_dict.get("val_f1"),
            1 - output_dict.get("val_fpr"),
        ]

        weights = SoftmaxWeightUpdater(init_theta=prev_theta)
        curr_E = weights.compute_energy(curr_metrics)
        weights.step(curr_metrics, prev_E)

        if curr_E:
            update = fl_sa(prev_E, curr_E, temp, output_folder, client)

            # ✅ REQUIRED ORDER: save_sa BEFORE count_update (your condition)
            save_sa(client, output_dict, temp, output_folder, weights, update=update)

            # ✅ count_update only touches the LAST entry
            if not update:
                count_update(output_folder, client, 1)

            return update
        else:
            return False

    return False


# weight updater
class SoftmaxWeightUpdater:
    def __init__(self, init_theta=None, lr=0.05):
        self.theta = np.array(init_theta) if init_theta is not None else np.zeros(3, dtype=float)
        self.lr = float(lr)

    def softmax(self):
        t = self.theta
        tmax = np.max(t)
        ex = np.exp(t - tmax)
        return ex / np.sum(ex)

    def compute_energy(self, metrics):
        w = self.softmax()
        return float(np.dot(w, metrics))

    def step(self, metrics, prev_E):
        metrics = np.asarray(metrics, dtype=float)
        weights_before = self.softmax()
        grad = weights_before * (metrics - prev_E)
        self.theta = self.theta - self.lr * grad

        
