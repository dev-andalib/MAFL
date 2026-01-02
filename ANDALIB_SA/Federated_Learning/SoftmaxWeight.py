from collections import Counter
import json
import os
import numpy as np
import math
import random
import torch

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
def save_sa(client_id, output_dict, temp, output_folder, weights, update=False):
    
    # update 
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if os.path.exists(output_file):
        existing_data = read_file(client_id, output_folder)

        if update:
            existing_data["val_accuracy"] = output_dict.get('val_accuracy')
            existing_data["val_f1"] = output_dict.get('val_f1')
            existing_data["val_fpr"] = output_dict.get('val_fpr')
            existing_data['temp'] = temp

            ### for energy ##
            curr_metrics = [output_dict.get('val_accuracy'), output_dict.get('val_f1'), 1-output_dict.get('val_fpr') ]
            curr_E = weights.compute_energy(curr_metrics)
            existing_data['prev_E'] = curr_E
            existing_data['theta'] = weights.theta.tolist()
            
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)

    #new save           
    else:
        ### for energy ##
        curr_metrics = [output_dict.get('val_accuracy'), output_dict.get('val_f1'), 1-output_dict.get('val_fpr') ]
        curr_E = weights.compute_energy(curr_metrics)
        client_data = {
            "val_accuracy": output_dict.get('val_accuracy'),
            "val_f1": output_dict.get('val_f1'),
            "val_fpr": output_dict.get('val_fpr'),
            "temp":temp,
            "prev_E": curr_E,
            "theta" : weights.theta.tolist(),
                       }
        with open(output_file, 'w') as f:
            json.dump(client_data, f, indent=4)






# SA send model updates or not
def file_handle(client, output_dict, temp):
    global output_folder 
    if type(client) == int or type(client) == str:
        
        if isFirst(client, output_folder): # file not created yet
                weights = SoftmaxWeightUpdater()
                save_sa(client, output_dict, temp, output_folder, weights) 
                return True  # Accept first client regardless
            
        else:
            existing_data = read_file(client, output_folder) # read acc
            if existing_data != None:
                prev_E = existing_data.get('prev_E')
                curr_metrics = [output_dict.get('val_accuracy'), output_dict.get('val_f1'), 1-output_dict.get('val_fpr') ] # get current E
                
                weights = SoftmaxWeightUpdater(init_theta=existing_data.get('theta'))
                curr_E = weights.compute_energy(curr_metrics)
                weights.step(curr_metrics, prev_E)
                
                if curr_E:
                    update = fl_sa(prev_E, curr_E, temp, output_folder, client) # SA below this function
                    if not update:
                        count_update(output_folder, client, 1)
                    save_sa(client, output_dict, temp, output_folder, weights,  update=update)
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
        





# weight updater
class SoftmaxWeightUpdater:
    def __init__(self, init_theta=None, lr=0.05): # initialize theta for weights [accuracy, f1, fpr]
        self.theta = np.array(init_theta) if init_theta is not None else np.zeros(3, dtype=float)
        self.lr = float(lr)

    def softmax(self): # for summing the weights to 1 through normalization
        t = self.theta
        tmax = np.max(t)
        ex = np.exp(t - tmax)
        return ex / np.sum(ex)

    def compute_energy(self, metrics): # compute energy based on weights and metrics
        w = self.softmax()
        return float(np.dot(w, metrics))

    def step(self, metrics, prev_E,): # update theta based on metrics and previous energy (as average energy)
        metrics = np.asarray(metrics, dtype=float)
        weights_before = self.softmax()
        grad = weights_before * (metrics - prev_E)
        self.theta = self.theta - self.lr * grad
        
        
        



# def print_result(path):
#     for i in range(10):
#         dictt = torch.load("E:/Tahmid_SA/Final Working SA/"+path+f"_{i}.pth")
#         print(f'Client {i}: ', dictt.keys())
#         print()    

# for i in ["best_multiclass_model", "best_binary_model", "best_joint_model"]:
#     print_result(i)