import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from collections import OrderedDict
from Federated_Learning.utility import file_handle, save_metrics_graphs
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset



# -------------------------------------------------------------------------
# 1. Get and set model weights
# -------------------------------------------------------------------------


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]



def set_weights(model, parameters):
    # Get the keys (layer names) from the model's state dictionary
    keys = model.state_dict().keys()
    
    # Create an ordered dictionary mapping keys to the new parameters (converted to Tensors)
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    # Load the new state dictionary into the model
    model.load_state_dict(state_dict, strict=True)


# -------------------------------------------------------------------------
# 1. DATA LOAD, PARTITION & PREPROCESS
# -------------------------------------------------------------------------
def create_sequences(X_data, y_data, seq_len):
    """Helper to create time-series sequences."""
    if hasattr(X_data, 'values'): X_data = X_data.values
    if hasattr(y_data, 'values'): y_data = y_data.values
    
    num_samples = len(X_data) - seq_len + 1
    
    # Handle edge case where partition is smaller than sequence length
    if num_samples <= 0:
        return torch.FloatTensor([]), torch.FloatTensor([])

    X_seq = np.zeros((num_samples, seq_len, X_data.shape[1]))
    y_seq = np.zeros(num_samples)

    for i in range(num_samples):
        X_seq[i] = X_data[i:i+seq_len]
        y_seq[i] = y_data[i+seq_len-1]
        
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


def data_load_preprocess(partition_id, num_partitions, batch_size, seq_length=10, alpha=5):
    """
    Loads data, applies Dirichlet Partitioning (Non-IID), SMOTE (train only), 
    creates sequences, and returns DataLoaders.
    
    Args:
        alpha (float): Parameter for Dirichlet distribution. 
                       Lower value (e.g., 0.1) = High Heterogeneity (Non-IID).
                       Higher value (e.g., 100) = Low Heterogeneity (IID).
    """
    # 1. Load Data
    csv_path = "D:\T24\MAFL\ANDALIB_SA\Recent_UpdatedNB15.csv" # Fixed path string for Windows
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # 2. Encode Labels
    le = LabelEncoder()
    if 'is_sm_ips_ports' in df.columns:
        df['is_sm_ips_ports'] = le.fit_transform(df['is_sm_ips_ports'])
    df['label'] = le.fit_transform(df['label'])

    # 3. Feature Selection
    final_features = [
        'sttl', 'dttl', 'ct_state_ttl', 'rate', 'sload', 'dpkts',
        'dload', 'dinpkt', 'dur', 'proto', 'state', 'dmean',
        'sbytes', 'sinpkt', 'sjit', 'ct_dst_sport_ltm', 'ct_srv_dst',
        'ct_src_dport_ltm', 'ackdat', 'synack'
    ]
    available_features = [f for f in final_features if f in df.columns]
    
    # Filter DF to relevant columns + label (needed for partitioning)
    df = df[available_features + ['label']]

    # ---------------------------------------------------------
    # 4. Partitioning Logic (Dirichlet Partitioner)
    # ---------------------------------------------------------
    
    # Convert Pandas DataFrame to HuggingFace Dataset
    # (Required by flwr_datasets partitioners)
    full_dataset = Dataset.from_pandas(df)

    # Configure Dirichlet Partitioner
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",    # Split based on class labels
        alpha=alpha,             # Controls degree of Non-IID
        min_partition_size=seq_length + 10, # Ensure partition isn't empty
        seed=42                  # CRITICAL: Ensures same split across all clients/runs
    )
    
    # Assign the dataset to the partitioner
    partitioner.dataset = full_dataset
    
    # Load the specific partition for this client
    client_dataset = partitioner.load_partition(partition_id)
    
    # Convert back to Pandas for processing
    df_partition = client_dataset.to_pandas()
    
    # Separate X and y
    X = df_partition[available_features]
    y = df_partition["label"]
    
    print(f"Partition {partition_id} size: {len(X)} | Distribution: {Counter(y)}")

    # ---------------------------------------------------------

    # 5. Split: Train (64%), Val (16%), Test (20%)
    # Use stratify=y to maintain the Dirichlet distribution in the splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
    )

    # 6. SMOTE (Train only)
    try:
        # Check if we have enough samples for SMOTE (k_neighbors=5 requires >= 6 samples)
        if len(X_train) > 6 and min(Counter(y_train).values()) > 5:
            smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, m_neighbors=10, kind='borderline-1')
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            print(f"Warning: Partition {partition_id} too small or disjoint for SMOTE. Using original data.")
            X_train_resampled, y_train_resampled = X_train, y_train
    except Exception as e:
        print(f"SMOTE failed for partition {partition_id}: {e}. Using original data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # 7. Create Sequences
    X_train_seq, y_train_seq = create_sequences(X_train_resampled, y_train_resampled, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

    # 8. DataLoaders
    # drop_last=False ensures we don't crash if the last batch is small
    train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=batch_size, shuffle=False)

    # 9. Pos Weight (for binary loss)
    # Handle cases where a partition might miss a class entirely (extreme Non-IID)
    binary_counts = Counter(y_train_seq.numpy())
    count_0 = binary_counts.get(0, 1) # Default to 1 to avoid ZeroDivision
    count_1 = binary_counts.get(1, 1)
    
    w0 = 1.0 / np.sqrt(count_0 / len(y_train_seq)) if len(y_train_seq) > 0 else 1.0
    w1 = 1.0 / np.sqrt(count_1 / len(y_train_seq)) if len(y_train_seq) > 0 else 1.0
    
    # Avoid division by zero if w0 is somehow 0
    pos_weight_val = (w1 / w0) if w0 > 0 else 1.0
    pos_weight = torch.FloatTensor([pos_weight_val])

    return train_loader, val_loader, test_loader, pos_weight




# -------------------------------------------------------------------------
# 2. MODEL CLASS
# -------------------------------------------------------------------------
class BinaryNIDS(nn.Module):
    def __init__(self, input_features=20, seq_length=10):
        super(BinaryNIDS, self).__init__()
        
        # CNN Layers
        self.conv1_3 = nn.Conv1d(input_features, 16, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(input_features, 8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(24, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM Layers
        self.bilstm = nn.LSTM(
            input_size=32, hidden_size=16, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 24), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def attention_net(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(lstm_output * weights, dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([self.conv1_3(x), self.conv1_5(x)], dim=1)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        return torch.sigmoid(self.classifier(self.attention_net(lstm_out)))





# -------------------------------------------------------------------------
# 3. TRAINING FUNCTION
# -------------------------------------------------------------------------
def train(model, train_loader, val_loader, device, epochs, learning_rate, pos_weight, cid, temp):
    """
    Trains the model, runs validation, and calculates acceptance logic.

    """
    results = {"train": {}, "val": {}}


    
    # Use BCELoss. If pos_weight is critical, consider BCEWithLogitsLoss and removing sigmoid in model.
    # Here keeping BCELoss as per original design.
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    model.train()
    model.to(device)

    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0

    
    # --- Training Loop ---
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            preds = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            # Binary Metrics
            train_tp += ((preds == 1) & (labels == 1)).sum().item()
            train_fp += ((preds == 1) & (labels == 0)).sum().item()
            train_fn += ((preds == 0) & (labels == 1)).sum().item()
            train_tn += ((preds == 0) & (labels == 0)).sum().item()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    # Calculate Training Metrics
    train_accuracy = train_correct / train_total if train_total > 0 else 0
    train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
    train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
    train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
    train_fpr = train_fp / (train_fp + train_tn) if (train_fp + train_tn) > 0 else 0
    
    results['train'] = { 
        "train_loss": train_loss.item(),
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_fpr": train_fpr
    }
    save_metrics_graphs(results['train'], cid, "train_metrics")





    # --- Validation Loop (for metrics & acceptance) ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).view(-1)
            val_loss += criterion(outputs, labels).item()
            
            preds = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            
            # Binary Metrics
            val_tp += ((preds == 1) & (labels == 1)).sum().item()
            val_fp += ((preds == 1) & (labels == 0)).sum().item()
            val_fn += ((preds == 0) & (labels == 1)).sum().item()
            val_tn += ((preds == 0) & (labels == 0)).sum().item()

    # Calculate Metrics
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
    val_fpr = val_fp / (val_fp + val_tn) if (val_fp + val_tn) > 0 else 0

    results['val'] = { 
        "val_loss": val_loss / len(val_loader),
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_fpr": val_fpr
    }
    
    # Logic for Client Acceptance (Placeholder: always accept)
    client_accept = file_handle(cid, results['val'], temp)
    save_metrics_graphs(results['val'], cid, "val_metrics")
    return results, client_accept









# -------------------------------------------------------------------------
# 4. TESTING FUNCTION
# -------------------------------------------------------------------------
def test(model, test_loader, device, cid):
    """
    Evaluates the model on the test set.
    """
    criterion = nn.BCELoss()
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).view(-1)
            running_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    result = {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_fpr": fpr
    }


    save_metrics_graphs(result, cid, "test_metrics")
    return avg_loss, total, result