from collections import OrderedDict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from datasets import load_dataset
from sklearn.feature_selection import SelectKBest, chi2
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import os
import json
from imblearn.over_sampling import SMOTE
from Federated_Learning.utility import file_handle

from Federated_Learning.helper import create_sequences, train_epoch_binary, train_epoch_joint, evaluate_binary, train_epoch_multiclass, evaluate_hierarchical, evaluate_multiclass, apply_weighted_smote, evaluate_joint

def freeze_multiclass_head(net):
    """Freeze multiclass head during binary training."""
    for param in net.multiclass_head.parameters():
        param.requires_grad = False
    print("Multiclass head frozen")

def freeze_binary_head(net):
    """Freeze binary head during multiclass training."""
    for param in net.binary_head.parameters():
        param.requires_grad = False
    print("Binary head frozen")

def unfreeze_all_heads(net):
    """Unfreeze all heads for joint training."""
    for param in net.binary_head.parameters():
        param.requires_grad = True
    for param in net.multiclass_head.parameters():
        param.requires_grad = True
    print("All heads unfrozen for joint training")

def get_trainable_params(net):
    """Get only trainable parameters for optimizer."""
    return [p for p in net.parameters() if p.requires_grad]

def load_preprocessing_info(filepath='preprocessing_info.json'):
    """
    Load preprocessing information from JSON file for consistent test data processing.
    
    Returns:
        dict: Contains top_20_features, label encoders info, and other preprocessing parameters
    """
    try:
        with open(filepath, 'r') as f:
            preprocessing_info = json.load(f)
        print(f"Loaded preprocessing information from {filepath}")
        return preprocessing_info
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Make sure to run training first to generate preprocessing info.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error reading {filepath}: {e}")
        return None

def apply_preprocessing_to_test_data(test_df, preprocessing_info):
    """
    Apply the same preprocessing steps to test data using saved preprocessing information.
    
    Args:
        test_df: DataFrame containing test data with all original features
        preprocessing_info: Dictionary containing preprocessing parameters
        
    Returns:
        tuple: (X_processed, y_binary, y_attack) - processed features and labels
    """
    if preprocessing_info is None:
        raise ValueError("Preprocessing info is required")
    
    # Extract features and labels
    X = test_df.drop(columns=["label", "attack_cat"])
    y_binary = test_df["label"]
    y_attack = test_df["attack_cat"]
    
    # Apply feature selection (use same top 20 features)
    top_20_features = preprocessing_info['top_20_features']
    X_selected = X[top_20_features]
    
    # Apply standard scaling using saved scaler parameters
    if 'scaler' in preprocessing_info:
        scaler_info = preprocessing_info['scaler']
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_info['mean'])
        scaler.scale_ = np.array(scaler_info['scale'])
        scaler.n_features_in_ = len(scaler_info['feature_names'])
        
        X_selected_scaled = scaler.transform(X_selected)
        X_selected = pd.DataFrame(X_selected_scaled, columns=X_selected.columns, index=X_selected.index)
        print("Applied saved StandardScaler to test data")
    
    # Apply label encoding using saved encoder info
    from sklearn.preprocessing import LabelEncoder
    
    # Check if 'is_sm_ips_ports' encoder is needed (only if feature is in selected features)
    if 'is_sm_ips_ports_encoder' in preprocessing_info and 'is_sm_ips_ports' in X_selected.columns:
        le_city = LabelEncoder()
        le_city.classes_ = np.array(preprocessing_info['is_sm_ips_ports_encoder']['classes'])
        X_selected['is_sm_ips_ports'] = le_city.transform(X_selected['is_sm_ips_ports'])
    
    # Reconstruct and apply GLOBAL attack encoder
    le_attack = LabelEncoder() 
    # Use global classes instead of saved classes to ensure consistency
    # 0 = Normal traffic, 1-9 = Different attack types
    all_attack_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    le_attack.fit(all_attack_classes)
    print(f"Global attack encoder classes for test data: {le_attack.classes_}")
    
    try:
        y_attack_encoded = le_attack.transform(y_attack)
        print("✓ Successfully applied global label encoding to test data")
    except ValueError as e:
        print(f"Test data label encoding error: {e}")
        print(f"Test attack classes found: {sorted(y_attack.unique())}")
        print(f"Expected attack classes: {all_attack_classes}")
        raise
    
    # Apply global attack mapping
    global_mapping = preprocessing_info['global_attack_mapping']
    y_attack_remapped = np.array([global_mapping.get(int(label), label) for label in y_attack_encoded])
    
    return X_selected, y_binary, y_attack_remapped

class UniformPartitioner:
    """
    Custom uniform partitioner that ensures all clients get every label equally distributed.
    This guarantees balanced representation across all federated learning clients.
    """
    
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.dataset = None
        self._partitions = {}
    
    def load_partition(self, partition_id: int):
        """Load a partition with uniform label distribution."""
        if self.dataset is None:
            raise ValueError("Dataset must be assigned before loading partitions")
        
        if partition_id not in self._partitions:
            self._create_uniform_partitions()
        
        return self._partitions[partition_id]
    
    def _create_uniform_partitions(self):
        """Create partitions ensuring uniform distribution of all labels across clients."""
        # Convert dataset to pandas for easier manipulation
        df = self.dataset.to_pandas()
        
        # Use actual column names from the dataset (label and attack_cat)
        binary_col = 'label'
        attack_col = 'attack_cat'
        
        # Verify columns exist
        if binary_col not in df.columns or attack_col not in df.columns:
            raise ValueError(f"Required columns '{binary_col}' and '{attack_col}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Get unique labels for both binary and multiclass
        unique_binary = df[binary_col].unique()
        unique_attacks = df[attack_col].unique()
        
        print(f"Creating uniform partitions for {self.num_partitions} clients")
        print(f"Binary labels: {unique_binary}")
        print(f"Attack labels: {unique_attacks}")
        
        # Initialize partitions
        partitions = [[] for _ in range(self.num_partitions)]
        
        # For each unique combination of binary and attack labels
        for binary_label in unique_binary:
            for attack_label in unique_attacks:
                # Get all samples with this label combination
                mask = (df[binary_col] == binary_label) & (df[attack_col] == attack_label)
                samples = df[mask]
                
                if len(samples) == 0:
                    continue
                
                # Shuffle samples for random distribution
                samples = samples.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Distribute samples equally across all partitions
                samples_per_partition = len(samples) // self.num_partitions
                remainder = len(samples) % self.num_partitions
                
                start_idx = 0
                for partition_id in range(self.num_partitions):
                    # Calculate end index - distribute remainder samples to first few partitions
                    extra = 1 if partition_id < remainder else 0
                    end_idx = start_idx + samples_per_partition + extra
                    
                    # Add samples to partition
                    partition_samples = samples.iloc[start_idx:end_idx]
                    partitions[partition_id].append(partition_samples)
                    
                    start_idx = end_idx
        
        # Combine samples for each partition and convert back to dataset format
        for partition_id in range(self.num_partitions):
            if partitions[partition_id]:
                partition_df = pd.concat(partitions[partition_id], ignore_index=True)
                # Shuffle the final partition
                partition_df = partition_df.sample(frac=1, random_state=42 + partition_id).reset_index(drop=True)
            else:
                # Empty partition - shouldn't happen with proper data
                partition_df = pd.DataFrame(columns=df.columns)
            
            # Convert back to dataset format
            from datasets import Dataset
            self._partitions[partition_id] = Dataset.from_pandas(partition_df)
            
            # Print distribution statistics using actual column names
            binary_dist = partition_df[binary_col].value_counts().sort_index()
            attack_dist = partition_df[attack_col].value_counts().sort_index()
            print(f"Partition {partition_id}: {len(partition_df)} samples")
            print(f"  Binary distribution: {dict(binary_dist)}")
            print(f"  Attack distribution: {dict(attack_dist)}")

class Net(nn.Module):
    def __init__(self, input_features=20, seq_length=10, num_attack_types=9):  # APPROACH 1: Clean 9-class model (0-8)
        super().__init__()
        
        # ANALYSIS: CNN-BiLSTM Architecture for Network Intrusion Detection
        # Input: Sequential network traffic data [batch, seq_length, features]
        # Output: Binary classification (normal/attack) + Multiclass (9 attack types: 0-8)
        # APPROACH 1: No wasted neurons - all 9 classes used efficiently
        
        # Multi-scale CNN feature extraction
        self.conv1_3 = nn.Conv1d(input_features, 16, kernel_size=3, padding=1)  # Short patterns
        self.conv1_5 = nn.Conv1d(input_features, 8, kernel_size=5, padding=2)   # Long patterns
        
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(24, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        
        self.multiclass_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_attack_types)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def attention_net(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
        return weighted_output
    
    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        
        conv_3 = self.conv1_3(x)
        conv_5 = self.conv1_5(x)
        x = torch.cat([conv_3, conv_5], dim=1)
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        
        features = self.attention_net(lstm_out)
        
        return features
    
    def forward(self, x, stage='both'):
        features = self.extract_features(x)
        
        # Hierarchical inference: Binary first, then multiclass only if attack detected
        if stage == 'binary':
            binary_output = self.binary_head(features)
            return binary_output
        elif stage == 'multiclass':
            multiclass_output = self.multiclass_head(features)
            return multiclass_output
        elif stage == 'hierarchical':
            # Hierarchical inference: only compute multiclass for detected attacks
            binary_output = self.binary_head(features)
            binary_probs = torch.sigmoid(binary_output.squeeze())
            attack_mask = binary_probs > 0.5  # Attack detected
            
            # Initialize multiclass output with zeros - APPROACH 1: 9 classes (0-8)
            multiclass_output = torch.zeros(binary_output.size(0), 9, device=binary_output.device)
            if attack_mask.any():
                # Only compute multiclass for detected attacks
                multiclass_output[attack_mask] = self.multiclass_head(features[attack_mask])
            
            return binary_output, multiclass_output
        else:  # stage == 'both' (for joint training)
            binary_output = self.binary_head(features)
            multiclass_output = self.multiclass_head(features)
            return binary_output, multiclass_output
 
def get_weights(net):
    try:
        # Ensure all tensors are moved to CPU and synchronized before extracting weights
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        weights = []
        for _, val in net.state_dict().items():
            if val.is_cuda:
                # Clear any pending CUDA operations before moving to CPU
                torch.cuda.synchronize()
                weights.append(val.cpu().numpy())
            else:
                weights.append(val.numpy())
        return weights
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in get_weights: {e}")
            # Force CPU mode and retry
            net = net.cpu()
            return [val.numpy() for _, val in net.state_dict().items()]
        else:
            raise e
 
def set_weights(net, parameters):
    """
    FEDERATED LEARNING WEIGHT SETTING:
    - Handles potential architecture mismatches gracefully
    - Ensures consistent model structure across clients
    - Provides detailed error reporting for debugging
    """
    try:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # FIXED: Add shape verification before loading
        net_state_dict = net.state_dict()
        for key, param in state_dict.items():
            if key in net_state_dict:
                if param.shape != net_state_dict[key].shape:
                    print(f"WARNING: Shape mismatch for {key}: expected {net_state_dict[key].shape}, got {param.shape}")
                    
                    # Handle CNN input layer mismatch (conv1_3, conv1_5)
                    if ("conv1_3" in key or "conv1_5" in key) and "weight" in key:
                        target_shape = net_state_dict[key].shape  # [out_channels, in_channels, kernel_size]
                        if param.shape[1] != target_shape[1]:  # Input channel mismatch
                            print(f"FIXING CNN input channels: Adjusting {key} from {param.shape} to {target_shape}")
                            out_channels, old_in_channels, kernel_size = param.shape
                            _, new_in_channels, _ = target_shape
                            
                            if old_in_channels < new_in_channels:
                                # Pad with zeros to increase input channels
                                padding = torch.zeros(out_channels, new_in_channels - old_in_channels, kernel_size)
                                param = torch.cat([param, padding], dim=1)
                                print(f"  Padded input channels from {old_in_channels} to {new_in_channels}")
                            else:
                                # Truncate to reduce input channels
                                param = param[:, :new_in_channels, :]
                                print(f"  Truncated input channels from {old_in_channels} to {new_in_channels}")
                            state_dict[key] = param
                    
                    # Handle multiclass head mismatch 
                    elif "multiclass_head" in key and len(param.shape) >= 1:
                        target_shape = net_state_dict[key].shape
                        if param.shape[0] != target_shape[0]:  # Output dimension mismatch
                            print(f"FIXING: Adjusting {key} from {param.shape} to {target_shape}")
                            if len(param.shape) == 1:  # Bias
                                if param.shape[0] > target_shape[0]:
                                    param = param[:target_shape[0]]  # Truncate
                                else:
                                    padding = torch.zeros(target_shape[0] - param.shape[0])
                                    param = torch.cat([param, padding])  # Pad
                            elif len(param.shape) == 2:  # Weight
                                if param.shape[0] > target_shape[0]:
                                    param = param[:target_shape[0], :]  # Truncate rows
                                else:
                                    padding = torch.zeros(target_shape[0] - param.shape[0], param.shape[1])
                                    param = torch.cat([param, padding], dim=0)  # Pad rows
                            state_dict[key] = param
                    
                    # Handle other layer mismatches
                    else:
                        print(f"  Skipping layer {key} due to incompatible shapes")
                        # Use client's current layer weights instead of server's
                        state_dict[key] = net_state_dict[key].clone()
        
        net.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully")
        
    except Exception as e:
        print(f"ERROR in set_weights: {e}")
        print("Attempting fallback weight loading...")
        try:
            # Fallback: load weights with strict=False
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)
            print("Fallback weight loading successful (some layers may be uninitialized)")
        except Exception as fallback_error:
            print(f"CRITICAL ERROR: Both strict and non-strict loading failed: {fallback_error}")
            raise fallback_error
 
fds = None  # Cache FederatedDataset
path = "local_data.csv"  # Local data path
meta = None
def load_data(partition_id: int, num_partitions: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Custom data division mechanism to divide data into partitions, selecting top 20 features using chi-squared test.
    Returns train, validation, and test DataLoaders.
    """
    global path, fds, meta, meta_path
    # FIXED: Don't override the batch_size parameter
    test_size = 0.3
    random_state = 42
    
    if fds is None:
        ds = load_dataset("csv", data_files=path, split="train")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = ds
        fds = partitioner
 
    # Load the partitioned dataset
    df  = fds.load_partition(partition_id).with_format("pandas")[:]
    


    X = df.drop(columns=["label", "attack_cat"])
    y_binary = df["label"]
    y_attack = df["attack_cat"]

    # --- Use predefined top 20 features ---
    top_20_features = [
        "sttl",
        "ct_state_ttl", 
        "state",
        "dur",
        "dload",
        "sload",
        "dmean",
        "sinpkt",
        "ct_dst_sport_ltm",
        "spkts",
        "rate",
        "sloss",
        "dpkts",
        "ct_src_dport_ltm",
        "sbytes",
        "dloss",
        "dbytes",
        "smean",
        "proto",
        "ct_dst_src_ltm"
    ]
    
    # Filter X to only include top 20 features
    X = X[top_20_features]

    # --- Split data ---
    X_train, X_temp, y_binary_train, y_binary_temp, y_attack_train, y_attack_temp = train_test_split(
        X, y_binary, y_attack,
        test_size=test_size, stratify=y_binary, random_state=random_state
    )

    X_val, X_test, y_binary_val, y_binary_test, y_attack_val, y_attack_test = train_test_split(
        X_temp, y_binary_temp, y_attack_temp,
        test_size=0.5, stratify=y_binary_temp, random_state=random_state
    )
    
    # --- Apply StandardScaler to features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to maintain column names for later processing
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Applied StandardScaler to features")
    
    # GLOBAL label encoder for attack categories that knows ALL possible attack classes
    # This ensures consistency across all clients in federated learning
    le_attack = LabelEncoder()
    
    # Define ALL possible attack classes that can appear in the dataset (0-9)
    # 0 = Normal traffic, 1-9 = Different attack types
    all_attack_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    le_attack.fit(all_attack_classes)  # Fit on ALL possible classes, not just partition data
    
    print(f"Global attack encoder classes: {le_attack.classes_}")
    
    # Debug: Show original attack class distribution
    print(f"BEFORE encoding - attack_cat unique values:")
    print(f"  Train: {sorted(y_attack_train.unique())}")
    print(f"  Val: {sorted(y_attack_val.unique())}")
    print(f"  Test: {sorted(y_attack_test.unique())}")
    
    # Now safely transform all splits - this won't fail even if some classes are missing
    try:
        y_attack_train = le_attack.transform(y_attack_train)
        y_attack_val = le_attack.transform(y_attack_val) 
        y_attack_test = le_attack.transform(y_attack_test)
        print("✓ Successfully applied global label encoding to all splits")
        
        # Debug: Show encoded attack class distribution
        print(f"AFTER encoding - attack_cat unique values:")
        print(f"  Train: {sorted(np.unique(y_attack_train))}")
        print(f"  Val: {sorted(np.unique(y_attack_val))}")
        print(f"  Test: {sorted(np.unique(y_attack_test))}")
        
    except ValueError as e:
        print(f"Label encoding error: {e}")
        print(f"Training attack classes: {sorted(y_attack_train.unique())}")
        print(f"Validation attack classes: {sorted(y_attack_val.unique())}")
        print(f"Test attack classes: {sorted(y_attack_test.unique())}")
        raise
    
    # --- Save preprocessing information to JSON ---
    # Define global attack mapping (will be used later in the function)
    # This maps the label-encoded attack classes to the model's class range (0-8)
    # Label encoder preserves original values: 0=Normal, 1-9=Attack types
    # For multiclass model: map attack classes 1-9 to 0-8
    GLOBAL_ATTACK_MAPPING = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8
        # Note: Class 0 (Normal) should not appear in attack-only data and is not mapped
    }
    
    preprocessing_info = {
        'top_20_features': top_20_features,
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_names': top_20_features
        },
        'attack_encoder': {
            'classes': le_attack.classes_.tolist(),
            'is_global_encoder': True,
            'all_possible_attack_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
        'global_attack_mapping': GLOBAL_ATTACK_MAPPING,
        'feature_selection_method': 'predefined_features',
        'num_selected_features': 20,
        'data_preprocessing_order': ['predefined_feature_selection', 'train_test_split', 'standard_scaling', 'label_encoding', 'multiclass_smote', 'binary_smote', 'sequencing'],
        'sequence_length': 10,
        'test_size': 0.3,
        'random_state': 42
    }
    
    # Save to JSON file
    preprocessing_file = 'preprocessing_info.json'
    with open(preprocessing_file, 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    print(f"Preprocessing information saved to {preprocessing_file}")

    # --- Convert DataFrames to numpy arrays first, then to tensors ---
    X_train = torch.from_numpy(X_train.values).float()  # Convert DataFrame to numpy then tensor
    X_val = torch.from_numpy(X_val.values).float()
    X_test = torch.from_numpy(X_test.values).float()

    y_binary_train = torch.from_numpy(y_binary_train.values).float()
    y_binary_val = torch.from_numpy(y_binary_val.values).float()
    y_binary_test = torch.from_numpy(y_binary_test.values).float()

    y_attack_train = torch.from_numpy(y_attack_train).long()
    y_attack_val = torch.from_numpy(y_attack_val).long()
    y_attack_test = torch.from_numpy(y_attack_test).long()

    # --- Filter attack-only subset ---
    attack_indices = (y_binary_train == 1)
    y_attack_train_filtered = y_attack_train[attack_indices]
    X_train_attacks = X_train[attack_indices]
    
    # Debug: Check consistency between binary and attack labels
    print(f"Binary label distribution: {torch.bincount(y_binary_train.long())}")
    print(f"Attack samples (label=1): {attack_indices.sum()} out of {len(y_binary_train)}")
    if len(y_attack_train_filtered) > 0:
        print(f"Attack-only subset unique classes: {sorted(torch.unique(y_attack_train_filtered).tolist())}")
        
        # Check for data inconsistency: attack samples with attack_cat=0
        if 0 in y_attack_train_filtered:
            print("⚠️  INCONSISTENCY DETECTED: Found samples with label=1 (attack) but attack_cat=0 (normal)")
            inconsistent_count = (y_attack_train_filtered == 0).sum()
            print(f"   Number of inconsistent samples: {inconsistent_count}")
            # Remove inconsistent samples
            valid_attack_mask = y_attack_train_filtered != 0
            y_attack_train_filtered = y_attack_train_filtered[valid_attack_mask]
            X_train_attacks = X_train_attacks[valid_attack_mask]
            print(f"   After removing inconsistent samples: {len(y_attack_train_filtered)} valid attack samples")

    if len(y_attack_train_filtered) == 0:
        print("Warning: No valid attack samples found in training data after filtering!")
        # Create dummy data with a valid attack class (encoded 1 = original class 1)
        y_attack_train_filtered = torch.LongTensor([1])  # Use encoded class 1 instead of 0
        X_train_attacks = X_train[:1]

    unique_attacks = torch.unique(y_attack_train_filtered)

    # APPROACH 1: Clean 9-class model architecture for federated learning
    # Solution: Eliminate wasted class 0, use efficient 0-8 mapping
    
    print(f"Original unique attacks: {unique_attacks}")
    print("APPROACH 1: Proper hierarchical prediction with clean 9-class architecture (0-8)")
    
    # FIX 1: Map attack labels to ensure they fit in 0-9 range
    unique_attacks_list = unique_attacks.tolist()
    print(f"Unique attack labels found: {unique_attacks_list}")
    
    # CRITICAL FIX: Use GLOBAL consistent mapping across ALL clients to ensure model compatibility
    # This ensures all clients map the same original labels to the same new labels
    # Based on the most common attack types in UNSW-NB15 dataset
    print("Using GLOBAL consistent attack label mapping for federated learning compatibility")
    
    # Use the global mapping defined earlier for preprocessing info
    print(f"GLOBAL attack label mapping (consistent across all clients): {GLOBAL_ATTACK_MAPPING}")
    attack_to_new_label = GLOBAL_ATTACK_MAPPING
    
    # Apply remapping
    y_attack_train_remapped = y_attack_train.clone()
    y_attack_val_remapped = y_attack_val.clone() 
    y_attack_test_remapped = y_attack_test.clone()
    
    for old_label, new_label in attack_to_new_label.items():
        mask_train = y_attack_train == old_label
        mask_val = y_attack_val == old_label
        mask_test = y_attack_test == old_label
        
        y_attack_train_remapped[mask_train] = new_label
        y_attack_val_remapped[mask_val] = new_label
        y_attack_test_remapped[mask_test] = new_label
        
    # For filtered subset, apply remapping to the already filtered attack labels
    y_attack_train_filtered_remapped = y_attack_train_filtered.clone()
    
    # Validation: Ensure no Normal labels (0) in attack-only subset
    if torch.any(y_attack_train_filtered == 0):
        print("⚠️  Still found Normal labels (0) in attack-only data after initial filtering!")
        print(f"Attack-only labels before remapping: {torch.unique(y_attack_train_filtered)}")
        # Remove any remaining Normal labels
        valid_mask = y_attack_train_filtered != 0
        if valid_mask.sum() > 0:
            y_attack_train_filtered = y_attack_train_filtered[valid_mask]
            print(f"Removed {(~valid_mask).sum()} Normal samples from attack-only data")
        else:
            print("⚠️  All attack samples were Normal! Using dummy attack class.")
            y_attack_train_filtered = torch.LongTensor([1])  # Use attack class 1
    
    # Apply remapping to convert attack classes to model range (0-8)
    for old_label, new_label in attack_to_new_label.items():
        mask = y_attack_train_filtered == old_label
        y_attack_train_filtered_remapped[mask] = new_label
    
    print(f"Final attack labels range: {torch.unique(y_attack_train_filtered_remapped)}")
    print("APPROACH 1: Clean 9-class model architecture - no wasted neurons")

    # FIX 2: Proper binary class weights
    num_attacks = (y_binary_train == 1).sum().float()
    num_normal = (y_binary_train == 0).sum().float()
    pos_weight = torch.tensor([num_normal / torch.clamp(num_attacks, min=1.0)])
    print(f"Binary pos_weight: {pos_weight.item():.4f} (normal/attack ratio)")

    # Show ORIGINAL class distribution (before SMOTE for diagnostic purposes)
    attack_only_labels = y_attack_train_filtered_remapped
    original_attack_counts = Counter(attack_only_labels.numpy())
    actual_num_classes = 9  # Fixed 9 classes for clean model (0-8)
    print(f"APPROACH 1: Using {actual_num_classes} classes")
    print(f"ORIGINAL (pre-SMOTE) attack class distribution: {dict(sorted(original_attack_counts.items()))}")
    
    # Show imbalance before SMOTE
    if original_attack_counts:
        min_samples = min(original_attack_counts.values())
        max_samples = max(original_attack_counts.values())
        imbalance_ratio = max_samples / max(min_samples, 1)
        print(f"ORIGINAL imbalance ratio: {imbalance_ratio:.2f}x (will be fixed by SMOTE)")
    
    # Note: Final attack_class_weights will be calculated AFTER SMOTE
    # in the apply_weighted_smote function based on the balanced distribution
    
    # Class imbalance will be handled by SMOTE and weights calculated after SMOTE
    print("Class imbalance will be handled by SMOTE + weighted loss (calculated after SMOTE)")

    # CRITICAL FIX: Extract attack samples BEFORE sequencing for SMOTE
    print("STEP 1: Extract attack samples from 2D data for SMOTE processing...")
    
    # --- Filter attack-only subset from 2D data (before sequencing) ---
    attack_indices = (y_binary_train == 1)
    y_attack_train_filtered_remapped = y_attack_train_remapped[attack_indices]
    X_train_attacks_2d = X_train[attack_indices]  # 2D data for SMOTE
    
    # Get attack samples from validation set (2D)
    val_attack_mask = y_binary_val == 1
    X_multi_val_2d = X_val[val_attack_mask]  # 2D validation attack samples
    y_multi_val = y_attack_val_remapped[val_attack_mask]  # Their attack labels
    
    print(f"Attack samples extracted: Train={X_train_attacks_2d.shape}, Val={X_multi_val_2d.shape}")
    
    # STEP 2: Apply SMOTE to 2D attack data
    print("STEP 2: Applying SMOTE to 2D attack data...")
    if len(X_train_attacks_2d) > 0:
        # Check which attack classes are present in this client's data
        present_classes = torch.unique(y_attack_train_filtered_remapped).tolist()
        missing_classes = [i for i in range(actual_num_classes) if i not in present_classes]
        
        if missing_classes:
            print(f"⚠️  Client missing attack classes: {missing_classes}")
            print(f"   Present attack classes: {present_classes}")
            print("   This is normal in federated learning with many clients")
        
        X_multi_train_2d, y_multi_train, attack_class_weights = apply_weighted_smote(
            X_train_attacks_2d, y_attack_train_filtered_remapped, actual_num_classes
        )
        
        # Ensure class weights tensor is properly sized (9 classes)
        if len(attack_class_weights) != actual_num_classes:
            print(f"⚠️  Adjusting class weights from {len(attack_class_weights)} to {actual_num_classes} classes")
            full_class_weights = torch.ones(actual_num_classes)
            # Copy available weights
            for i in range(min(len(attack_class_weights), actual_num_classes)):
                full_class_weights[i] = attack_class_weights[i]
            attack_class_weights = full_class_weights
        
        print(f"POST-SMOTE: Updated attack_class_weights calculated from balanced data: {attack_class_weights}")
        print("✓ Class weights are now based on SMOTE-balanced distribution, not original imbalanced data")
    else:
        print("No attack samples found for SMOTE - client has only normal traffic")
        X_multi_train_2d = X_train_attacks_2d
        y_multi_train = y_attack_train_filtered_remapped
        attack_class_weights = torch.ones(actual_num_classes)
    
    # STEP 2B: Apply SMOTE to binary classification data (normal vs attack)
    print("STEP 2B: Applying SMOTE to binary classification data...")
    
    # Show original binary class distribution
    original_binary_counts = Counter(y_binary_train.numpy())
    print(f"ORIGINAL binary class distribution: Normal={original_binary_counts.get(0, 0)}, Attack={original_binary_counts.get(1, 0)}")
    
    # Apply SMOTE to balance binary classes
    from imblearn.over_sampling import SMOTE
    binary_smote = SMOTE(random_state=42, k_neighbors=min(5, len(X_train) - 1))
    
    try:
        # Custom SMOTE implementation that tracks indices for proper attack label mapping
        from imblearn.over_sampling import SMOTE
        
        # Create a custom class to track SMOTE indices
        class SMOTEWithIndices(SMOTE):
            def _fit_resample(self, X, y):
                X_resampled, y_resampled = super()._fit_resample(X, y)
                # Get sample indices - original samples first, then synthetic
                n_original = len(X)
                self.sample_indices_ = list(range(n_original))
                
                # For synthetic samples, track which original sample they were generated from
                n_synthetic = len(X_resampled) - n_original
                minority_class = 1  # attack class
                minority_indices = np.where(y == minority_class)[0]
                
                # SMOTE generates synthetic samples from minority class
                # We need to distribute them among minority samples
                samples_per_minority = n_synthetic // len(minority_indices) if len(minority_indices) > 0 else 0
                remainder = n_synthetic % len(minority_indices) if len(minority_indices) > 0 else 0
                
                for i, minority_idx in enumerate(minority_indices):
                    # Add base samples for this minority sample
                    for _ in range(samples_per_minority):
                        self.sample_indices_.append(minority_idx)
                    # Add remainder samples to first few minority samples
                    if i < remainder:
                        self.sample_indices_.append(minority_idx)
                
                return X_resampled, y_resampled
        
        # Apply SMOTE with index tracking
        smote_with_indices = SMOTEWithIndices(random_state=42, k_neighbors=min(5, len(X_train) - 1))
        X_train_binary_smote, y_binary_train_smote = smote_with_indices.fit_resample(
            X_train.numpy(), y_binary_train.numpy()
        )
        
        # Map attack labels using tracked indices
        y_attack_train_smote = []
        for sample_idx in smote_with_indices.sample_indices_:
            if sample_idx < len(y_attack_train_remapped):  # Original sample
                y_attack_train_smote.append(y_attack_train_remapped[sample_idx].item())
            else:  # This shouldn't happen with our logic, but safety check
                y_attack_train_smote.append(0)  # Default to normal
        
        y_attack_train_smote = np.array(y_attack_train_smote)
        
        # Update training data with SMOTE results
        X_train = torch.from_numpy(X_train_binary_smote).float()
        y_binary_train = torch.from_numpy(y_binary_train_smote.astype(float)).float()
        y_attack_train_remapped = torch.from_numpy(y_attack_train_smote).long()
        
        # Show post-SMOTE binary distribution
        smote_binary_counts = Counter(y_binary_train_smote)
        print(f"POST-SMOTE binary class distribution: Normal={smote_binary_counts.get(0, 0)}, Attack={smote_binary_counts.get(1, 0)}")
        print(f"✓ Binary SMOTE applied successfully - balanced binary classes for main training")
        
        # Recalculate binary pos_weight after SMOTE (should be close to 1.0 now)
        num_attacks_smote = (y_binary_train == 1).sum().float()
        num_normal_smote = (y_binary_train == 0).sum().float()
        pos_weight = torch.tensor([num_normal_smote / torch.clamp(num_attacks_smote, min=1.0)])
        print(f"Updated binary pos_weight after SMOTE: {pos_weight.item():.4f}")
        
    except Exception as e:
        print(f"Binary SMOTE failed: {e}. Using original imbalanced data.")
        # Keep original data if SMOTE fails
    
    # STEP 3: Apply sequencing to ALL datasets (main + SMOTE-processed multiclass)
    print("STEP 3: Applying sequencing to all datasets...")
    seq_length = 10
    
    # Create sequences for main training data (now SMOTE-balanced)
    X_train_seq, y_binary_train_seq, y_attack_train_seq = create_sequences(
        X_train.numpy(), y_binary_train.numpy(), y_attack_train_remapped.numpy(), seq_length=seq_length
    )
    
    # Create sequences for validation data  
    X_val_seq, y_binary_val_seq, y_attack_val_seq = create_sequences(
        X_val.numpy(), y_binary_val.numpy(), y_attack_val_remapped.numpy(), seq_length=seq_length
    )
    
    # Create sequences for test data
    X_test_seq, y_binary_test_seq, y_attack_test_seq = create_sequences(
        X_test.numpy(), y_binary_test.numpy(), y_attack_test_remapped.numpy(), seq_length=seq_length
    )
    
    # Create sequences for SMOTE-processed multiclass training data
    if len(X_multi_train_2d) > 0:
        X_multi_train_seq, _, y_multi_train_seq = create_sequences(
            X_multi_train_2d.numpy(), 
            y_multi_train.numpy(),  # dummy binary labels (not used)
            y_multi_train.numpy(),  # actual multiclass labels
            seq_length=seq_length
        )
        X_multi_train = torch.from_numpy(X_multi_train_seq).float()
        y_multi_train = torch.from_numpy(y_multi_train_seq).long()
        print(f"Sequenced multiclass training data shape: {X_multi_train.shape}")
    else:
        # Initialize empty tensors if no attack data
        X_multi_train = torch.zeros((0, seq_length, 20))  
        y_multi_train = torch.zeros(0, dtype=torch.long)
    
    # Create sequences for multiclass validation data  
    if len(X_multi_val_2d) > 0:
        X_multi_val_seq, _, y_multi_val_seq = create_sequences(
            X_multi_val_2d.numpy(),
            y_multi_val.numpy(),  # dummy binary labels
            y_multi_val.numpy(),  # actual multiclass labels  
            seq_length=seq_length
        )
        X_multi_val = torch.from_numpy(X_multi_val_seq).float()
        y_multi_val = torch.from_numpy(y_multi_val_seq).long()
        print(f"Sequenced multiclass validation data shape: {X_multi_val.shape}")
    else:
        # Initialize empty tensors if no attack data
        X_multi_val = torch.zeros((0, seq_length, 20))
        y_multi_val = torch.zeros(0, dtype=torch.long)
    
    # Convert main datasets back to tensors
    X_train = torch.from_numpy(X_train_seq).float()
    y_binary_train = torch.from_numpy(y_binary_train_seq).float()  # Fixed: float for BCEWithLogitsLoss
    y_attack_train_remapped = torch.from_numpy(y_attack_train_seq).long()
    
    X_val = torch.from_numpy(X_val_seq).float()
    y_binary_val = torch.from_numpy(y_binary_val_seq).float()  # Fixed: float for BCEWithLogitsLoss
    y_attack_val_remapped = torch.from_numpy(y_attack_val_seq).long()
    
    X_test = torch.from_numpy(X_test_seq).float()
    y_binary_test = torch.from_numpy(y_binary_test_seq).float()  # Fixed: float for BCEWithLogitsLoss
    y_attack_test_remapped = torch.from_numpy(y_attack_test_seq).long()
    
    print(f"All sequenced data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # CRITICAL: Detect actual number of input features from sequenced data
    actual_input_features = X_train.shape[2]  # [batch, seq_length, features]
    print(f"DETECTED: Actual input features = {actual_input_features}")

    # --- Datasets & loaders ---
    train_dataset = TensorDataset(X_train, y_binary_train, y_attack_train_remapped)
    val_dataset   = TensorDataset(X_val,   y_binary_val,   y_attack_val_remapped)
    test_dataset  = TensorDataset(X_test,  y_binary_test,  y_attack_test_remapped)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # STEP 4: Create multiclass data loaders from sequenced data
    print("STEP 4: Creating multiclass data loaders from sequenced data...")
    
    if len(X_multi_train) > 0:
        print(f"Multiclass training samples: {len(X_multi_train)}")
        print(f"Multiclass validation samples: {len(X_multi_val)}")
        print(f"Multiclass validation classes: {torch.unique(y_multi_val)}")
        
        # Create multiclass datasets from already processed (SMOTE + sequenced) data
        multiclass_dataset = TensorDataset(X_multi_train, y_multi_train)
        
        # Handle multiclass validation set
        if len(X_multi_val) > 0:
            multiclass_val_dataset = TensorDataset(X_multi_val, y_multi_val)
            multiclass_val_loader = DataLoader(multiclass_val_dataset, batch_size=batch_size, shuffle=False)
            print(f"Created multiclass validation loader with {len(X_multi_val)} samples")
        else:
            # Fallback: use a small portion of training data for validation if no attacks in val set
            print("WARNING: No attack samples in validation set, using portion of training for validation")
            val_size = min(len(X_multi_train) // 5, 100)  # Use 20% or max 100 samples
            multiclass_val_dataset = TensorDataset(X_multi_train[:val_size], y_multi_train[:val_size])
            multiclass_val_loader = DataLoader(multiclass_val_dataset, batch_size=batch_size, shuffle=False)
            
        multiclass_loader = DataLoader(multiclass_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"✓ Successfully created multiclass loaders with SMOTE-balanced and sequenced data")
        
    else:
        # No attack samples found - create empty loaders
        print("WARNING: No attack samples found for multiclass training")
        empty_X = torch.zeros((1, 10, 20))  # Minimal tensor with correct shape
        empty_y = torch.zeros(1, dtype=torch.long)
        multiclass_dataset = TensorDataset(empty_X, empty_y)
        multiclass_loader = DataLoader(multiclass_dataset, batch_size=batch_size, shuffle=True)
        multiclass_val_loader = DataLoader(multiclass_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader, val_loader, test_loader,
        multiclass_loader, multiclass_val_loader, pos_weight, attack_class_weights
    )

 
def train(net, trainloader, valloader, multiclass_loader, multiclass_val_loader, pos_weight, attack_class_weights, epochs, learning_rate, device, temp, cid):
    """
    HIERARCHICAL TRAINING ANALYSIS:
    Phase 1: Binary classification (Normal vs Attack)
    Phase 2: Multiclass classification (Attack type identification) - ONLY on attack samples
    Phase 3: Joint fine-tuning (Both tasks together)
    
    POTENTIAL ISSUES TO MONITOR:
    1. Data imbalance between normal/attack samples
    2. Inconsistent class distributions across federated clients
    3. Model efficiency (eliminated wasted class - now 9 classes)
    4. Loss function weight balancing
    5. Feature extraction freezing/unfreezing logic
    """
    print(f"Training started on client {cid} using device: {device}")
    
    net.to(device)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # FIX 2: Add multiclass class weights to handle imbalance
    # Create balanced class weights for multiclass classification
    multiclass_criterion = nn.CrossEntropyLoss(weight=attack_class_weights.to(device))
    
    print(f"Client {cid}: Using IMPROVED CrossEntropy with class weights for multiclass")

    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Phase 1: Binary Classification Training
    print(f"Client {cid}: Phase 1 - Training Binary Classifier")
    
    # Freeze multiclass head to prevent interference
    freeze_multiclass_head(net)
    
    # Create optimizer with only trainable parameters
    optimizer = torch.optim.Adam(get_trainable_params(net), lr=0.001)
    
    num_epochs = 10  # Updated: Set to 20 epochs
    patience = 3  # Early stopping patience
    best_val_loss = float('inf')
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        try:
            train_loss, train_acc = train_epoch_binary(net, trainloader, binary_criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_binary(net, valloader, binary_criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models', exist_ok=True)
                torch.save(net.state_dict(), f'models/best_binary_model_{cid}.pth')
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Client {cid}: Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Client {cid} - Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                print(f"No improvement count: {no_improve_count}/{patience}")
                
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            break
    
    # Load best binary model
    best_model_path = f'models/best_binary_model_{cid}.pth'
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # Phase 2: Multi-class Classification Training (freeze feature extractors)
    print(f"Client {cid}: Phase 2 - Training Multi-class Classifier")
    
    # Check if we have enough attack samples for multiclass training
    if len(multiclass_loader.dataset) == 0:
        print(f"Client {cid}: No attack samples found, skipping multiclass training")
        multi_train_losses = []
        multi_train_accs = []
    else:
        # Get actual number of unique attack classes for this client
        unique_attack_classes = torch.unique(multiclass_loader.dataset.tensors[1])
        num_unique_classes = len(unique_attack_classes)
        print(f"Client {cid}: Training multiclass with {num_unique_classes} attack types: {unique_attack_classes}")
        
        # BASIC: Simple multiclass training - no freezing, no complexity
        print(f"Client {cid}: Training multiclass with {num_unique_classes} attack types: {unique_attack_classes}")
        print(f"Client {cid}: Using BASIC approach - no layer freezing, no advanced techniques")
        
        # Verify model architecture is correct
        model_multiclass_size = net.multiclass_head[-1].out_features
        print(f"Client {cid}: Model multiclass head size: {model_multiclass_size}")
        assert model_multiclass_size == 9, f"Model should have 9 classes (Approach 1), got {model_multiclass_size}"
        
        # Freeze binary head to prevent degradation of Phase 1 performance
        freeze_binary_head(net)
        print(f"Client {cid}: Binary head frozen - training only multiclass head and features")
        
        # Create optimizer with only trainable parameters
        optimizer_multi = torch.optim.Adam(get_trainable_params(net), lr=0.001)
        
        multi_train_losses = []
        multi_train_accs = []
        
        # Updated: Set epochs to 20 and add early stopping
        multiclass_epochs = 10  # Updated: Set to 20 epochs
        multiclass_patience = 3  # Early stopping patience
        best_multi_loss = float('inf')
        multi_no_improve_count = 0
        print(f"Client {cid}: Starting {multiclass_epochs} epochs of multiclass training with early stopping")
        
        for epoch in range(multiclass_epochs):
            try:
                train_loss, train_acc = train_epoch_multiclass(net, multiclass_loader, multiclass_criterion, optimizer_multi, device)
                
                # FIX 3: Use validation loss for early stopping instead of training loss
                val_loss, val_acc = evaluate_multiclass(net, multiclass_val_loader, multiclass_criterion, device)
                
                multi_train_losses.append(train_loss)
                multi_train_accs.append(train_acc)
                
                # Early stopping logic for multiclass using VALIDATION loss
                if val_loss < best_multi_loss:
                    best_multi_loss = val_loss
                    multi_no_improve_count = 0
                    os.makedirs('models', exist_ok=True)
                    torch.save(net.state_dict(), f'models/best_multiclass_model_{cid}.pth')
                else:
                    multi_no_improve_count += 1
                    if multi_no_improve_count >= multiclass_patience:
                        print(f"Client {cid}: Early stopping multiclass at epoch {epoch+1} (no improvement for {multiclass_patience} epochs)")
                        break
                
                if (epoch + 1) % 5 == 0:
                    print(f"Client {cid} - Multiclass Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    print(f"No improvement count: {multi_no_improve_count}/{multiclass_patience}")
                    
            except Exception as e:
                print(f"Error in multiclass epoch {epoch}: {e}")
                break
        
        # Load best multiclass model if it exists
        best_multiclass_path = f'models/best_multiclass_model_{cid}.pth'
        if os.path.exists(best_multiclass_path):
            net.load_state_dict(torch.load(best_multiclass_path, map_location=device, weights_only=True))
            print(f"Client {cid}: Loaded best multiclass model")
    
    # Phase 3: Joint Fine-tuning - with early stopping
    print(f"Client {cid}: Phase 3 - Joint Fine-tuning with Early Stopping")
    
    # Unfreeze all heads for joint training
    unfreeze_all_heads(net)
    
    # Create optimizer with all parameters now trainable
    optimizer_joint = torch.optim.Adam(net.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    
    joint_train_losses = []
    joint_val_losses = []
    joint_train_accs = []
    joint_val_accs = []
    
    # Updated: Set epochs to 20 and add early stopping
    joint_epochs = 10  # Updated: Set to 20 epochs
    joint_patience = 3  # Early stopping patience
    best_joint_loss = float('inf')
    joint_no_improve_count = 0
    
    for epoch in range(joint_epochs):
        try:
            train_loss, train_acc = train_epoch_joint(net, trainloader, binary_criterion, multiclass_criterion, optimizer_joint, device)
            val_loss, val_avg_acc, val_avg_rec, val_avg_prec, val_avg_fpr, val_macro_f1 = evaluate_joint(net, valloader, multiclass_val_loader, binary_criterion, multiclass_criterion, device)
            
            joint_train_losses.append(train_loss)
            joint_val_losses.append(val_loss)
            joint_train_accs.append(train_acc)
            joint_val_accs.append(val_avg_acc)  # Track averaged accuracy for consistency
            
            # Early stopping logic for joint training
            if val_loss < best_joint_loss:
                best_joint_loss = val_loss
                joint_no_improve_count = 0
                os.makedirs('models', exist_ok=True)
                torch.save(net.state_dict(), f'models/best_joint_model_{cid}.pth')
            else:
                joint_no_improve_count += 1
                if joint_no_improve_count >= joint_patience:
                    print(f"Client {cid}: Early stopping joint training at epoch {epoch+1} (no improvement for {joint_patience} epochs)")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Client {cid} - Joint Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Binary Acc: {train_acc:.4f}")
                print(f"Val Avg Acc: {val_avg_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}, Val Avg FPR: {val_avg_fpr:.4f}")
                print(f"No improvement count: {joint_no_improve_count}/{joint_patience}")
                
        except Exception as e:
            print(f"Error in joint epoch {epoch}: {e}")
            break
    
    # Load best joint model if it exists
    best_joint_path = f'models/best_joint_model_{cid}.pth'
    if os.path.exists(best_joint_path):
        net.load_state_dict(torch.load(best_joint_path, map_location=device, weights_only=True))
        print(f"Client {cid}: Loaded best joint model")
    
     # Return comprehensive metrics from all training phases
    final_metrics = {
        "val_loss": val_loss if 'val_loss' in locals() else 0.0,
        "val_accuracy": val_avg_acc if 'val_avg_acc' in locals() else 0.0,
        "val_precision": val_avg_prec if 'val_avg_prec' in locals() else 0.0,
        "val_recall": val_avg_rec if 'val_avg_rec' in locals() else 0.0,
        "val_f1": val_macro_f1 if 'val_macro_f1' in locals() else 0.0,
        "val_fpr": val_avg_fpr if 'val_avg_fpr' in locals() else 0.0,
        "binary_training_complete": True,
        "multiclass_training_complete": len(multi_train_losses) > 0 if 'multi_train_losses' in locals() else False,
        "joint_training_complete": len(joint_train_losses) > 0 if 'joint_train_losses' in locals() else False,
        "final_train_loss": joint_train_losses[-1] if 'joint_train_losses' in locals() and joint_train_losses else train_losses[-1] if train_losses else 0.0,
        "final_val_loss": joint_val_losses[-1] if 'joint_val_losses' in locals() and joint_val_losses else val_losses[-1] if val_losses else 0.0,
        # Early stopping info
        "early_stop_binary": no_improve_count >= patience if 'no_improve_count' in locals() else False,
        "early_stop_multiclass": multi_no_improve_count >= multiclass_patience if 'multi_no_improve_count' in locals() else False,
        "early_stop_joint": joint_no_improve_count >= joint_patience if 'joint_no_improve_count' in locals() else False
    }

    print(temp)
    client_accept = file_handle(cid,  final_metrics, temp)
    print(f"Client {cid}: All 3 phases of training completed successfully!")
    return final_metrics, client_accept
 
def test(net, testloader, device, cid=None):
    """Validate the model on the test set."""
    print(f"Testing client {cid}...")
    print(f"Test loader length: {len(testloader)}")
    print(f"Test dataset size: {len(testloader.dataset) if hasattr(testloader, 'dataset') else 'Unknown'}")
    
    net = net.to(device)
    
    # Initialize default values
    total_loss = 0.0
    binary_acc = 0.0
    multi_acc = 0.0
    
    try:
        # Create criteria for evaluation
        binary_criterion = nn.BCEWithLogitsLoss()
        multiclass_criterion = nn.CrossEntropyLoss()
        
        total_loss, binary_acc, multi_acc, binary_preds, multi_preds, true_binary, true_attack = evaluate_hierarchical(
            net, testloader, binary_criterion, multiclass_criterion, device, hierarchical=True
        )
        
        # Calculate precision and recall metrics
        # Binary classification metrics
        binary_precision = precision_score(true_binary, binary_preds, average='binary', zero_division=0)
        binary_recall = recall_score(true_binary, binary_preds, average='binary', zero_division=0)
        
        # Multiclass metrics (only on attack samples where binary correctly detected attacks)
        attack_mask = (true_binary == 1) & (binary_preds == 1)
        if attack_mask.sum() > 0:
            multi_precision = precision_score(true_attack[attack_mask], multi_preds[attack_mask], average='weighted', zero_division=0)
            multi_recall = recall_score(true_attack[attack_mask], multi_preds[attack_mask], average='weighted', zero_division=0)
        else:
            multi_precision = 0.0
            multi_recall = 0.0
        
        print(f"Client {cid} - Final Test Results:")
        print(f"Binary Classification Accuracy: {binary_acc:.4f}")
        print(f"Binary Precision: {binary_precision:.4f}, Binary Recall: {binary_recall:.4f}")
        print(f"Multi-class Classification Accuracy (on detected attacks): {multi_acc:.4f}")
        print(f"Multi-class Precision: {multi_precision:.4f}, Multi-class Recall: {multi_recall:.4f}")
        print(f"Total Loss: {total_loss:.4f}")
    except Exception as e:
        print(f"Error in testing client {cid}: {e}")
        import traceback
        traceback.print_exc()
        # Values already initialized above
        binary_precision = 0.0
        binary_recall = 0.0
        multi_precision = 0.0
        multi_recall = 0.0
 
    output_dict = {
        "binary_acc": binary_acc,
        "multi_acc": multi_acc,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "multi_precision": multi_precision,
        "multi_recall": multi_recall,
    }
    
    print(f"Client {cid} returning: loss={total_loss}, test_size={len(testloader)}, output_dict={output_dict}")
    return total_loss, len(testloader), output_dict