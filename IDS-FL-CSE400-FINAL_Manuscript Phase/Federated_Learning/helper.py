from turtle import pd
import numpy as np
import torch
import json
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, confusion_matrix

def create_sequences(X, y_binary, y_attack_cat, seq_length=10):
    num_samples = len(X) - seq_length + 1
    # Use float32 to reduce memory usage
    X_seq = np.zeros((num_samples, seq_length, X.shape[1]), dtype=np.float32)
    y_binary_seq = np.zeros(num_samples, dtype=np.float32)
    y_attack_cat_seq = np.zeros(num_samples, dtype=np.int64)
    
    for i in range(num_samples):
        X_seq[i] = X[i:i+seq_length].astype(np.float32)
        y_binary_seq[i] = y_binary[i+seq_length-1]
        y_attack_cat_seq[i] = y_attack_cat[i+seq_length-1]
    
    return X_seq, y_binary_seq, y_attack_cat_seq

def train_epoch_joint(model, loader, binary_criterion, multiclass_criterion, optimizer, device):

    model.train()
    total_loss = 0
    binary_correct = 0
    total = 0
    multiclass_batches = 0

    # ✅ Added for F1 and FPR (binary, over all samples)
    tp = fp = tn = fn = 0
    
    for batch_x, batch_y_binary, batch_y_attack in loader:
        batch_x = batch_x.to(device)
        batch_y_binary = batch_y_binary.to(device)
        batch_y_attack = batch_y_attack.to(device)
        
        optimizer.zero_grad()
        
        binary_output, multiclass_output = model(batch_x, stage='both')
        binary_output = binary_output.squeeze()
        
        # Binary classification loss (always computed)
        binary_loss = binary_criterion(binary_output, batch_y_binary)
        
        # Multiclass loss (only for attack samples)
        attack_mask = batch_y_binary == 1
        if attack_mask.sum() > 0:
            attack_labels = batch_y_attack[attack_mask]
            attack_predictions = multiclass_output[attack_mask]
            multiclass_loss = multiclass_criterion(attack_predictions, attack_labels)
            # ENHANCED: Adaptive loss balancing - increase multiclass weight over time
            epoch_progress = min(1.0, multiclass_batches / 1000)  # Gradually increase weight
            multiclass_weight = 0.2 + 0.3 * epoch_progress  # Start at 0.2, increase to 0.5
            total_loss_batch = binary_loss + multiclass_weight * multiclass_loss
            multiclass_batches += 1
        else:
            total_loss_batch = binary_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
        predicted = (binary_output > 0.0).float()  # Use 0.0 for logits
        binary_correct += (predicted == batch_y_binary).sum().item()
        total += batch_y_binary.size(0)

        # ✅ Added: confusion elements for F1/FPR (same scope as accuracy)
        tp += ((predicted == 1) & (batch_y_binary == 1)).sum().item()
        fp += ((predicted == 1) & (batch_y_binary == 0)).sum().item()
        tn += ((predicted == 0) & (batch_y_binary == 0)).sum().item()
        fn += ((predicted == 0) & (batch_y_binary == 1)).sum().item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = binary_correct / total

    # ✅ Added: F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ✅ Added: FPR
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return avg_loss, avg_acc, f1, fpr




def train_epoch_binary(model, loader, criterion, optimizer, device):
    """
    BINARY CLASSIFICATION TRAINING ANALYSIS:
    - Uses BCEWithLogitsLoss (expects raw logits, not sigmoid)
    - Threshold at 0.0 for logits (equivalent to 0.5 for probabilities)
    - Critical: Model should NOT apply sigmoid in forward pass
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # NEW: collect predictions + labels for F1/FPR
    all_preds = []
    all_true = []
    
    for batch_x, batch_y_binary, _ in loader:
        batch_x, batch_y_binary = batch_x.to(device), batch_y_binary.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, stage='binary').squeeze()
        loss = criterion(outputs, batch_y_binary)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Threshold at 0 for logits
        predicted = (outputs > 0.0).float()
        correct += (predicted == batch_y_binary).sum().item()
        total += batch_y_binary.size(0)

        # NEW: store for metrics
        all_preds.extend(predicted.detach().cpu().numpy().astype(int).tolist())
        all_true.extend(batch_y_binary.detach().cpu().numpy().astype(int).tolist())

    # NEW: F1
    f1 = f1_score(all_true, all_preds, average="binary", zero_division=0)

    # NEW: FPR = FP / (FP + TN)
    tn = sum((t == 0 and p == 0) for t, p in zip(all_true, all_preds))
    fp = sum((t == 0 and p == 1) for t, p in zip(all_true, all_preds))
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return total_loss / len(loader), correct / total, f1, fpr



def evaluate_binary(model, loader, criterion, device):
    """
    BINARY EVALUATION ANALYSIS:
    - Consistent threshold logic with training (0.0 for logits)
    - Proper precision/recall calculation for imbalanced datasets
    - F1-score as primary metric for attack detection
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for batch_x, batch_y_binary, _ in loader:
            batch_x, batch_y_binary = batch_x.to(device), batch_y_binary.to(device)
            
            outputs = model(batch_x, stage='binary').squeeze()
            loss = criterion(outputs, batch_y_binary)
            
            total_loss += loss.item()
            # FIXED: Consistent threshold logic with training
            predicted = (outputs > 0.0).float()  # Use 0.0 threshold for logits
            correct += (predicted == batch_y_binary).sum().item()
            total += batch_y_binary.size(0)
            
            # Calculate confusion matrix elements
            tp += ((predicted == 1) & (batch_y_binary == 1)).sum().item()
            fp += ((predicted == 1) & (batch_y_binary == 0)).sum().item()
            tn += ((predicted == 0) & (batch_y_binary == 0)).sum().item()
            fn += ((predicted == 0) & (batch_y_binary == 1)).sum().item()
    
    accuracy = correct / total
    
    # FIXED: Robust precision/recall calculation for edge cases
    if tp + fn == 0:  # No actual attacks in data
        recall = 1.0 if tp + fp == 0 else 0.0  # Perfect if no predictions either
    else:
        recall = tp / (tp + fn)
        
    if tp + fp == 0:  # No attack predictions
        precision = 1.0 if tp + fn == 0 else 0.0  # Perfect if no actual attacks
    else:
        precision = tp / (tp + fp)
    
    # F1-score calculation with edge case handling
    if precision + recall == 0:
        f1 = 0.0  # Both precision and recall are 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # ✅ Added: False Positive Rate (FPR)
    if fp + tn == 0:
        fpr = 0.0
    else:
        fpr = fp / (fp + tn)
    
    
    return total_loss / len(loader), accuracy, f1, fpr

def train_epoch_multiclass(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_true = []
    all_pred = []
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, stage='multiclass')
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

        # collect for metrics
        all_true.append(batch_y.detach().cpu())
        all_pred.append(predicted.detach().cpu())

    # --- added metrics (single averaged values) ---
    all_true = torch.cat(all_true).numpy()
    all_pred = torch.cat(all_pred).numpy()

    multi_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(all_true, all_pred)
    num_classes = cm.shape[0]
    fpr_list = []
    for i in range(num_classes):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[:, i].sum() - cm[i, :].sum() + cm[i, i]
        fpr_list.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
    multi_fpr = float(sum(fpr_list) / len(fpr_list)) if len(fpr_list) > 0 else 0.0
    # --------------------------------------------

    return total_loss / len(loader), correct / total, multi_f1, multi_fpr


def evaluate_multiclass(model, loader, criterion, device):
    """Evaluate multiclass classifier on attack samples only."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch_x, batch_y_attack in loader:
            batch_x = batch_x.to(device)
            batch_y_attack = batch_y_attack.to(device)
            
            # Only get multiclass output
            multiclass_output = model(batch_x, stage='multiclass')
            
            loss = criterion(multiclass_output, batch_y_attack)
            total_loss += loss.item()
            
            _, predicted = multiclass_output.max(1)
            total += batch_y_attack.size(0)
            correct += predicted.eq(batch_y_attack).sum().item()

            # collect for metrics
            all_true.append(batch_y_attack.detach().cpu())
            all_pred.append(predicted.detach().cpu())
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    # ---- added averaged multiclass metrics ----
    if total > 0:
        all_true = torch.cat(all_true).numpy()
        all_pred = torch.cat(all_pred).numpy()

        # Weighted F1 (single scalar)
        f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

        # Macro-averaged FPR
        cm = confusion_matrix(all_true, all_pred)
        num_classes = cm.shape[0]
        fpr_list = []
        for i in range(num_classes):
            FP = cm[:, i].sum() - cm[i, i]
            TN = cm.sum() - cm[:, i].sum() - cm[i, :].sum() + cm[i, i]
            fpr_list.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)

        fpr = sum(fpr_list) / len(fpr_list) if len(fpr_list) > 0 else 0.0
    else:
        f1 = 0.0
        fpr = 0.0
    # -----------------------------------------

    return avg_loss, accuracy, f1, fpr

def evaluate_hierarchical(model, loader, binary_criterion, multiclass_criterion, device, hierarchical=False):
    """
    HIERARCHICAL EVALUATION ANALYSIS:
    Critical Issues Found:
    1. Inconsistent threshold logic between binary training/evaluation
    2. Poor multiclass performance due to class imbalance
    3. Hierarchical prediction logic needs improvement
    
    Fixes Applied:
    - Consistent 0.0 threshold for logits
    - Better handling of edge cases
    - Improved hierarchical decision making
    - Added loss calculation
    """
    model.eval()
    binary_predictions = []
    multiclass_predictions = []
    true_binary = []
    true_attack = []
    total_binary_loss = 0
    total_multiclass_loss = 0
    multiclass_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y_binary, batch_y_attack in loader:
            batch_x = batch_x.to(device)
            batch_y_binary = batch_y_binary.to(device)
            batch_y_attack = batch_y_attack.to(device)
            
            # Use hierarchical inference if specified
            if hierarchical:
                binary_output, multiclass_output = model(batch_x, stage='hierarchical')
            else:
                binary_output, multiclass_output = model(batch_x, stage='both')
            
            # Calculate binary loss
            binary_loss = binary_criterion(binary_output.squeeze(), batch_y_binary)
            total_binary_loss += binary_loss.item()
            
            # Calculate multiclass loss (only for attack samples)
            attack_mask = batch_y_binary == 1
            if attack_mask.sum() > 0:
                attack_labels = batch_y_attack[attack_mask]
                attack_predictions = multiclass_output[attack_mask]
                multiclass_loss = multiclass_criterion(attack_predictions, attack_labels)
                total_multiclass_loss += multiclass_loss.item()
                multiclass_batches += 1
            
            # FIXED: Consistent threshold logic
            binary_pred = (binary_output.squeeze() > 0.0).float()  # Use 0.0 for logits
            _, multiclass_raw_pred = multiclass_output.max(1)
            
            # IMPROVED HIERARCHICAL PREDICTION LOGIC:
            # Step 1: Binary classifier decides normal vs attack
            # Step 2: If attack predicted, multiclass classifier determines attack type
            # Step 3: If normal predicted, force multiclass to 0 (normal)
            multiclass_pred = torch.where(
                binary_pred == 0,  # If binary predicts normal
                torch.zeros_like(multiclass_raw_pred),  # Force multiclass to 0 (normal)
                multiclass_raw_pred  # Otherwise use multiclass prediction
            )
            
            binary_predictions.extend(binary_pred.cpu().numpy())
            multiclass_predictions.extend(multiclass_pred.cpu().numpy())
            true_binary.extend(batch_y_binary.cpu().numpy())
            true_attack.extend(batch_y_attack.cpu().numpy())
    
    binary_predictions = np.array(binary_predictions)
    multiclass_predictions = np.array(multiclass_predictions)
    true_binary = np.array(true_binary)
    true_attack = np.array(true_attack)
    
    # Calculate average losses
    avg_binary_loss = total_binary_loss / len(loader)
    avg_multiclass_loss = total_multiclass_loss / max(multiclass_batches, 1)
    total_loss = avg_binary_loss + avg_multiclass_loss
    
    # Binary classification accuracy
    binary_accuracy = np.mean(binary_predictions == true_binary)
    
    # IMPROVED: Multiclass accuracy calculation
    # Only evaluate on samples where:
    # 1. Binary correctly identified as attack (true positive cases)
    # 2. AND ground truth is actually attack
    correct_attack_detection = (binary_predictions == 1) & (true_binary == 1)
    
    if correct_attack_detection.sum() > 0:
        # Calculate accuracy only on correctly detected attacks
        multiclass_accuracy = np.mean(
            multiclass_predictions[correct_attack_detection] == true_attack[correct_attack_detection]
        )
        print(f"Multiclass evaluated on {correct_attack_detection.sum()} correctly detected attacks")
    else:
        multiclass_accuracy = 0.0
        print("No correctly detected attacks for multiclass evaluation")
    
    # Additional diagnostics
    total_attacks = np.sum(true_binary == 1)
    detected_attacks = np.sum(binary_predictions == 1)
    print(f"Total attacks: {total_attacks}, Detected: {detected_attacks}")
    print(f"Binary recall: {np.sum(correct_attack_detection) / max(total_attacks, 1):.4f}")
    print(f"Binary loss: {avg_binary_loss:.4f}, Multiclass loss: {avg_multiclass_loss:.4f}")
    
    return total_loss, binary_accuracy, multiclass_accuracy, binary_predictions, multiclass_predictions, true_binary, true_attack

def evaluate_joint(model, trainloader, multiclass_val_loader, binary_criterion, multiclass_criterion, device):
    """
    Joint evaluation for joint training validation.
    Evaluates both binary and multiclass performance to guide joint training.
    
    Args:
        model: The neural network model
        trainloader: DataLoader for binary evaluation (contains normal + attack samples)
        multiclass_val_loader: DataLoader for multiclass evaluation (attack samples only)
        binary_criterion: Loss function for binary classification
        multiclass_criterion: Loss function for multiclass classification
        device: Device to run evaluation on
        
    Returns:
        combined_loss: Combined loss (binary + multiclass)
        avg_accuracy: Average of binary and multiclass accuracy
        avg_recall: Average of binary and multiclass recall
        avg_precision: Average of binary and multiclass precision
        avg_fpr: Average of binary and multiclass FPR
        macro_f1: Macro average of binary and multiclass F1 scores
    """
    model.eval()
    
    # Binary evaluation on validation set
    with torch.no_grad():
        total_binary_loss = 0
        binary_correct = 0
        binary_total = 0
        tp = fp = tn = fn = 0
        
        # Evaluate binary classification
        for batch_x, batch_y_binary, _ in trainloader:
            batch_x, batch_y_binary = batch_x.to(device), batch_y_binary.to(device)
            
            binary_output = model(batch_x, stage='binary').squeeze()
            binary_loss = binary_criterion(binary_output, batch_y_binary)
            
            total_binary_loss += binary_loss.item()
            predicted = (binary_output > 0.0).float()
            binary_correct += (predicted == batch_y_binary).sum().item()
            binary_total += batch_y_binary.size(0)
            
            # Calculate confusion matrix elements
            tp += ((predicted == 1) & (batch_y_binary == 1)).sum().item()
            fp += ((predicted == 1) & (batch_y_binary == 0)).sum().item()
            tn += ((predicted == 0) & (batch_y_binary == 0)).sum().item()
            fn += ((predicted == 0) & (batch_y_binary == 1)).sum().item()
        
        # Calculate binary metrics
        binary_accuracy = binary_correct / binary_total
        
        # Robust precision/recall calculation
        if tp + fn == 0:
            binary_recall = 1.0 if tp + fp == 0 else 0.0
        else:
            binary_recall = tp / (tp + fn)
            
        if tp + fp == 0:
            binary_precision = 1.0 if tp + fn == 0 else 0.0
        else:
            binary_precision = tp / (tp + fp)
        
        # Calculate binary FPR
        if fp + tn == 0:
            binary_fpr = 0.0
        else:
            binary_fpr = fp / (fp + tn)
        
        if binary_precision + binary_recall == 0:
            binary_f1 = 0.0
        else:
            binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall)
        
        avg_binary_loss = total_binary_loss / len(trainloader)
        
        # Multiclass evaluation on attack samples only
        total_multiclass_loss = 0
        multiclass_correct = 0
        multiclass_total = 0
        multiclass_tp = multiclass_fp = multiclass_tn = multiclass_fn = 0
        
        if len(multiclass_val_loader.dataset) > 0:
            for batch_x, batch_y_attack in multiclass_val_loader:
                batch_x = batch_x.to(device)
                batch_y_attack = batch_y_attack.to(device)
                
                multiclass_output = model(batch_x, stage='multiclass')
                multiclass_loss = multiclass_criterion(multiclass_output, batch_y_attack)
                
                total_multiclass_loss += multiclass_loss.item()
                _, predicted = multiclass_output.max(1)
                multiclass_total += batch_y_attack.size(0)
                multiclass_correct += predicted.eq(batch_y_attack).sum().item()
                
                # Calculate multiclass confusion matrix elements (macro averaging approach)
                for class_id in torch.unique(torch.cat([batch_y_attack, predicted])):
                    class_id = class_id.item()
                    true_class = (batch_y_attack == class_id).float()
                    pred_class = (predicted == class_id).float()
                    
                    multiclass_tp += ((pred_class == 1) & (true_class == 1)).sum().item()
                    multiclass_fp += ((pred_class == 1) & (true_class == 0)).sum().item()
                    multiclass_tn += ((pred_class == 0) & (true_class == 0)).sum().item()
                    multiclass_fn += ((pred_class == 0) & (true_class == 1)).sum().item()
            
            multiclass_accuracy = multiclass_correct / multiclass_total if multiclass_total > 0 else 0
            avg_multiclass_loss = total_multiclass_loss / len(multiclass_val_loader)
            
            # Calculate multiclass precision, recall, FPR, F1
            if multiclass_tp + multiclass_fn == 0:
                multiclass_recall = 1.0 if multiclass_tp + multiclass_fp == 0 else 0.0
            else:
                multiclass_recall = multiclass_tp / (multiclass_tp + multiclass_fn)
                
            if multiclass_tp + multiclass_fp == 0:
                multiclass_precision = 1.0 if multiclass_tp + multiclass_fn == 0 else 0.0
            else:
                multiclass_precision = multiclass_tp / (multiclass_tp + multiclass_fp)
            
            if multiclass_fp + multiclass_tn == 0:
                multiclass_fpr = 0.0
            else:
                multiclass_fpr = multiclass_fp / (multiclass_fp + multiclass_tn)
            
            if multiclass_precision + multiclass_recall == 0:
                multiclass_f1 = 0.0
            else:
                multiclass_f1 = 2 * (multiclass_precision * multiclass_recall) / (multiclass_precision + multiclass_recall)
        else:
            multiclass_accuracy = 0.0
            avg_multiclass_loss = 0.0
            multiclass_precision = 0.0
            multiclass_recall = 0.0
            multiclass_fpr = 0.0
            multiclass_f1 = 0.0
        
        # Calculate averaged metrics
        avg_accuracy = (binary_accuracy + multiclass_accuracy) / 2
        avg_recall = (binary_recall + multiclass_recall) / 2
        avg_precision = (binary_precision + multiclass_precision) / 2
        avg_fpr = (binary_fpr + multiclass_fpr) / 2
        macro_f1 = (binary_f1 + multiclass_f1) / 2
        
        # Combined loss for early stopping (weighted combination)
        # Weight multiclass loss higher since it's typically harder to optimize
        combined_loss = avg_binary_loss + 1.5 * avg_multiclass_loss
        
        print(f"Joint Validation - Binary: Acc={binary_accuracy:.4f}, Prec={binary_precision:.4f}, Rec={binary_recall:.4f}, F1={binary_f1:.4f}, FPR={binary_fpr:.4f}")
        print(f"Joint Validation - Multiclass: Acc={multiclass_accuracy:.4f}, Prec={multiclass_precision:.4f}, Rec={multiclass_recall:.4f}, F1={multiclass_f1:.4f}, FPR={multiclass_fpr:.4f}")
        print(f"Joint Validation - Averaged: Acc={avg_accuracy:.4f}, Prec={avg_precision:.4f}, Rec={avg_recall:.4f}, Macro F1={macro_f1:.4f}, FPR={avg_fpr:.4f}")
        print(f"Joint Validation - Combined Loss: {combined_loss:.4f}")
        
    return combined_loss, avg_accuracy, avg_recall, avg_precision, avg_fpr, macro_f1

def apply_weighted_smote(X_train, y_train, actual_num_classes=9):
    """
    Apply weighted SMOTE to balance class distribution while preserving hierarchy.
    
    Args:
        X_train: Training features (numpy array or tensor)
        y_train: Training labels (numpy array or tensor)  
        actual_num_classes: Number of classes in the model (default: 9)
        
    Returns:
        X_train_balanced: SMOTE-balanced training features as torch tensor
        y_train_balanced: SMOTE-balanced training labels as torch tensor
        class_weights: Updated class weights as torch tensor
    """
    print("Applying Weighted SMOTE for hierarchy-preserving class balancing...")
    
    # Convert to numpy if tensors
    if torch.is_tensor(X_train):
        X_train_np = X_train.numpy()
    else:
        X_train_np = X_train
        
    if torch.is_tensor(y_train):
        y_train_np = y_train.numpy()
    else:
        y_train_np = y_train
    
    # Get class distribution
    attack_counts_train = Counter(y_train_np)
    print(f"Original class distribution: {dict(sorted(attack_counts_train.items()))}")
    
    # Check if SMOTE is applicable
    if len(attack_counts_train) <= 1:
        print("Insufficient classes for SMOTE (need at least 2 classes)")
        return torch.from_numpy(X_train_np).float(), torch.from_numpy(y_train_np).long(), torch.ones(actual_num_classes)
    
    # Check minimum class size
    min_class_size = min(attack_counts_train.values())
    if min_class_size < 2:
        print(f"Insufficient samples for SMOTE (minimum class has {min_class_size} samples, need at least 2)")
        return torch.from_numpy(X_train_np).float(), torch.from_numpy(y_train_np).long(), torch.ones(actual_num_classes)
    
    # Reshape data for SMOTE if needed (flatten sequences)
    original_shape = X_train_np.shape
    if len(original_shape) > 2:
        X_train_flat = X_train_np.reshape(X_train_np.shape[0], -1)
    else:
        X_train_flat = X_train_np
    
    print(f"Data shape for SMOTE: {X_train_flat.shape}")
    
    # Calculate weighted SMOTE strategy
    max_count = max(attack_counts_train.values())
    min_count = min(attack_counts_train.values())
    original_ratio = min_count / max_count
    
    # Target: Improve balance while preserving hierarchy
    desired_min_ratio = 0.20  # 20% - significant improvement
    
    print(f"Original imbalance ratio: {original_ratio:.3f}")
    print(f"Target ratio: {desired_min_ratio:.3f}")
    
    # Apply square root compression to reduce extreme gaps
    sqrt_counts = {class_id: np.sqrt(count) for class_id, count in attack_counts_train.items()}
    max_sqrt = max(sqrt_counts.values())
    min_sqrt = min(sqrt_counts.values())
    
    # Calculate weighted targets
    target_counts = {}
    sampling_strategy = {}
    
    for class_id, sqrt_val in sqrt_counts.items():
        # Normalize sqrt to 0-1 range
        if max_sqrt > min_sqrt:
            normalized = (sqrt_val - min_sqrt) / (max_sqrt - min_sqrt)
        else:
            normalized = 1.0
        
        # Scale to desired range
        desired_min_samples = int(max_count * desired_min_ratio)
        new_count = int(desired_min_samples + normalized * (max_count - desired_min_samples))
        target_counts[class_id] = new_count
        
        # Only add to sampling strategy if we need to increase samples
        original_count = attack_counts_train[class_id]
        if new_count > original_count and original_count >= 2:
            sampling_strategy[class_id] = new_count
    
    print(f"SMOTE sampling strategy: {sampling_strategy}")
    
    if not sampling_strategy:
        print("No classes need SMOTE augmentation")
        return torch.from_numpy(X_train_np).float(), torch.from_numpy(y_train_np).long(), torch.ones(actual_num_classes)
    
    # Apply SMOTE
    try:
        # Calculate k_neighbors based on smallest class
        smallest_class_count = min(attack_counts_train.values())
        k_neighbors = min(5, smallest_class_count - 1) if smallest_class_count > 1 else 1
        
        print(f"Using k_neighbors={k_neighbors} for SMOTE")
        
        # Try BorderlineSMOTE for better quality, fallback to standard SMOTE
        try:
            from imblearn.over_sampling import BorderlineSMOTE
            smote = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42,
                kind='borderline-1'
            )
            print("Using BorderlineSMOTE for higher quality synthetic samples")
        except ImportError:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )
            print("Using standard SMOTE")
        
        X_train_smoted, y_train_smoted = smote.fit_resample(X_train_flat, y_train_np)
        
        # Reshape back to original format if needed
        if len(original_shape) > 2:
            X_train_smoted = X_train_smoted.reshape(-1, original_shape[1], original_shape[2])
        
        synthetic_samples_added = X_train_smoted.shape[0] - X_train_flat.shape[0]
        improvement_ratio = X_train_smoted.shape[0] / X_train_flat.shape[0]
        
        print(f"SMOTE SUCCESS!")
        print(f"  Before: {X_train_flat.shape[0]} samples")
        print(f"  After:  {X_train_smoted.shape[0]} samples")
        print(f"  Added:  {synthetic_samples_added} synthetic samples ({improvement_ratio:.2f}x data)")
        
        # Show class balance improvement
        new_attack_counts = Counter(y_train_smoted)
        print(f"New class distribution: {dict(sorted(new_attack_counts.items()))}")
        
        # Calculate updated class weights (moderate weights since SMOTE balanced the data)
        new_attack_total = len(y_train_smoted)
        class_weights = torch.ones(actual_num_classes)
        
        for attack_label, count in new_attack_counts.items():
            if 0 <= attack_label < actual_num_classes:
                # Reduced weight scaling since SMOTE already provided balance
                raw_weight = new_attack_total / (len(new_attack_counts) * count)
                # Apply dampening factor to prevent over-conservative predictions
                dampening_factor = 0.6
                balanced_weight = 1.0 + (raw_weight - 1.0) * dampening_factor
                class_weights[attack_label] = balanced_weight
        
        print(f"Updated class weights: {class_weights}")
        
        # Convert back to tensors
        X_train_balanced = torch.from_numpy(X_train_smoted).float()
        y_train_balanced = torch.from_numpy(y_train_smoted).long()
        
        return X_train_balanced, y_train_balanced, class_weights
        
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Continuing with original training data...")
        return torch.from_numpy(X_train_np).float(), torch.from_numpy(y_train_np).long(), torch.ones(actual_num_classes)