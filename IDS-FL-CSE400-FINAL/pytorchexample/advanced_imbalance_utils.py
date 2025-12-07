import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing to prevent overconfident predictions on majority classes.
    Helps with generalization and reduces overfitting to frequent classes.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        loss = -(targets_smooth * log_preds).sum(dim=1).mean()
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    Focuses learning on hard-to-classify examples (typically rare classes).
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using effective number of samples.
    Especially effective for extreme class imbalance.
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights.to(inputs.device))
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def advanced_data_augmentation(features, targets, device='cpu'):
    """
    Advanced data augmentation for severe class imbalance.
    Implements SMOTE-like interpolation and noise addition.
    """
    # Move to CPU for processing
    features = features.cpu()
    targets = targets.cpu()
    
    # Calculate class distribution
    unique_classes, class_counts = torch.unique(targets, return_counts=True)
    class_counts_dict = {cls.item(): count.item() for cls, count in zip(unique_classes, class_counts)}
    
    # Calculate imbalance ratio
    max_count = max(class_counts_dict.values())
    min_count = min(class_counts_dict.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio <= 10:  # No severe imbalance
        return features.to(device), targets.to(device)
    
    # Find rare classes (< 500 samples or < 10% of max class)
    rare_threshold = min(500, max_count * 0.1)
    rare_classes = [cls for cls, count in class_counts_dict.items() if count < rare_threshold]
    
    augmented_features_list = []
    augmented_targets_list = []
    
    for rare_class in rare_classes:
        current_count = class_counts_dict[rare_class]
        print(f"ADVANCED augmentation for class {rare_class} ({current_count} samples)")
        
        # Find indices of rare class samples
        rare_indices = (targets == rare_class).nonzero(as_tuple=True)[0]
        
        if len(rare_indices) < 2:
            print(f"  Skipping class {rare_class}: insufficient samples for interpolation")
            continue
        
        rare_features = features[rare_indices]
        
        # Calculate target augmentation (aim for 10x original or max 5000)
        target_multiplier = min(10, 5000 // current_count) if current_count > 0 else 10
        target_size = current_count * target_multiplier
        augment_count = target_size - current_count
        
        print(f"  Target size: {target_size}, generating {augment_count} synthetic samples")
        
        for i in range(augment_count):
            # SMOTE-like interpolation: blend two random samples
            idx1, idx2 = torch.randperm(len(rare_indices))[:2]
            sample1 = rare_features[idx1]
            sample2 = rare_features[idx2]
            
            # Random interpolation weight
            alpha = torch.rand(1).item()
            interpolated = alpha * sample1 + (1 - alpha) * sample2
            
            # Add controlled gaussian noise for diversity
            noise_scale = 0.03  # 3% noise relative to std
            feature_std = interpolated.std(dim=0, keepdim=True)
            noise = torch.randn_like(interpolated) * noise_scale * feature_std
            augmented_sample = interpolated + noise
            
            # Apply small random scaling (0.95-1.05)
            scale_factor = 0.95 + 0.1 * torch.rand(1).item()
            augmented_sample = augmented_sample * scale_factor
            
            augmented_features_list.append(augmented_sample)
            augmented_targets_list.append(torch.tensor(rare_class, dtype=targets.dtype))
    
    if augmented_features_list:
        augmented_features = torch.stack(augmented_features_list)
        augmented_targets = torch.stack(augmented_targets_list)
        
        # Combine original and augmented data
        final_features = torch.cat([features, augmented_features], dim=0)
        final_targets = torch.cat([targets, augmented_targets], dim=0)
        
        print(f"ADVANCED augmentation: {len(features)} → {len(final_features)} samples")
        print(f"Generated {len(augmented_features_list)} synthetic samples total")
    else:
        final_features = features
        final_targets = targets
        print("No synthetic samples generated")
    
    return final_features.to(device), final_targets.to(device)

def progressive_rare_class_sampling(features, targets, epoch, max_epochs, device='cpu'):
    """
    Progressive sampling that gradually increases rare class representation.
    Early epochs focus on majority classes, later epochs emphasize rare classes.
    """
    # Calculate progress through training
    progress = epoch / max_epochs
    
    # Calculate class distribution
    unique_classes, class_counts = torch.unique(targets, return_counts=True)
    class_counts_dict = {cls.item(): count.item() for cls, count in zip(unique_classes, class_counts)}
    
    # Identify rare classes (bottom 30% by count)
    sorted_classes = sorted(class_counts_dict.items(), key=lambda x: x[1])
    rare_class_count = len(sorted_classes) // 3
    rare_classes = [cls for cls, _ in sorted_classes[:rare_class_count]]
    
    # Progressive sampling weight: start low, increase with epoch
    rare_class_boost = 1.0 + progress * 4.0  # 1x to 5x boost over training
    
    # Create sampling weights
    sampling_weights = torch.ones(len(targets))
    
    for rare_class in rare_classes:
        rare_mask = (targets == rare_class)
        sampling_weights[rare_mask] *= rare_class_boost
    
    # Normalize weights
    sampling_weights = sampling_weights / sampling_weights.sum() * len(targets)
    
    print(f"Epoch {epoch}: Progressive sampling with {rare_class_boost:.2f}x boost for rare classes")
    
    return sampling_weights.to(device)

def analyze_class_distribution(targets, class_names=None):
    """
    Analyze and print detailed class distribution statistics.
    """
    unique_classes, class_counts = torch.unique(targets, return_counts=True)
    total_samples = len(targets)
    
    print("\n=== CLASS DISTRIBUTION ANALYSIS ===")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Sort by count (descending)
    sorted_indices = torch.argsort(class_counts, descending=True)
    
    for idx in sorted_indices:
        cls = unique_classes[idx].item()
        count = class_counts[idx].item()
        percentage = (count / total_samples) * 100
        
        class_name = f"Class {cls}" if class_names is None else class_names.get(cls, f"Class {cls}")
        print(f"{class_name}: {count:5d} samples ({percentage:5.2f}%)")
    
    # Calculate imbalance metrics
    max_count = class_counts.max().item()
    min_count = class_counts.min().item()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 100:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("   Recommended: Advanced data augmentation + Focal/CB Loss")
    elif imbalance_ratio > 10:
        print("⚠️  Moderate class imbalance")
        print("   Recommended: Class weights + basic augmentation")
    else:
        print("✅ Relatively balanced dataset")
    
    print("=" * 40)
    
    # Convert to dict for return
    class_counts_dict = {unique_classes[i].item(): class_counts[i].item() for i in range(len(unique_classes))}
    
    return {
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts_dict,
        'total_samples': total_samples
    }
