import datetime
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from torch import nn


def get_subjects_labels(df, selected_columns):
    labels = []
    subjects = []

    for index, row in df[selected_columns].iterrows():
        for column in selected_columns:
            if pd.notna(row[column]):
                labels.append(column)
                subjects.append(row[column])

    subjects = np.array(subjects)
    labels = np.array(labels)
    return subjects, labels


def cosine_similarity(features):
    cos_sim = nn.CosineSimilarity(dim=2)

    f1 = features.unsqueeze(1)  # shape: [batch_size, 1, 32]
    f2 = features.unsqueeze(0)   # shape: [1, batch_size, 32]

    similarity_matrix = cos_sim(f1, f2)  # shape: [batch_size, batch_size]
    similarity_matrix = (similarity_matrix + 1) / 2  # [0, 1]

    return similarity_matrix


def calculate_metrics(all_labels, all_preds, all_probs=None):
    """
    Two Class metrics
    """
    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(all_labels, all_preds)

    metrics = {
        'acc': acc,
        'recall': recall,
        'spec': spec,
        'f1': f1,
        'cm': cm
    }

    if all_probs is not None:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)

    return metrics


def calculate_all_folds_avg_metrics(args, all_folds_metrics):
    avg_metrics = {
        'acc': 0.0,
        'spec': 0.0,
        'recall': 0.0,
        'auc': 0.0,
        'f1': 0.0
    }
    std_metrics = {}
    detailed_metrics = {metric: [] for metric in avg_metrics.keys()}

    # avg and std
    for metric in avg_metrics.keys():
        values = [fold_metrics[metric] for fold_metrics in all_folds_metrics.values()]
        avg_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
        detailed_metrics[metric] = values

    # numpy to list
    processed_fold_metrics = {}
    for fold, metrics in all_folds_metrics.items():
        processed_fold_metrics[fold] = {
            'acc': float(metrics['acc']),
            'recall': float(metrics['recall']),
            'spec': float(metrics['spec']),
            'auc': float(metrics['auc']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['cm'].tolist()
        }

    # result dict
    results = {
        'experiment_name': args.experiment_name,
        'class_names': args.class_names,
        'fold_metrics': processed_fold_metrics,
        'average_metrics': {k: float(v) for k, v in avg_metrics.items()},
        'std_metrics': {k: float(v) for k, v in std_metrics.items()},
        'detailed_metrics': detailed_metrics,
        'experiment_config': vars(args),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # save
    save_dir = os.path.join('results', args.class_names.replace(',', '_'))
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir,
        f'{args.experiment_name}_{args.class_names.replace(",", "_")}_{timestamp}.json'
    )

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {save_path}")

def set_seed(seed):
    set_determinism(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)