import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def plot_training_history(history, save_path="outputs/training_history.png"):
    """Saves professional visualization of learning curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', marker='s')
    ax1.set_title('Model Performance: Accuracy History', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss Plot
    ax2.plot(history.history['loss'], label='Training Loss', color='#d62728', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#9467bd', marker='s')
    ax2.set_title('Model Performance: Loss History', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] 📊 Clinical History plot saved to '{save_path}'")

def evaluate_model(model, test_generator, save_dir="outputs"):
    """
    Generates industry-standard evaluation metrics.
    Includes Confusion Matrix and ROC-AUC Curve.
    """
    if test_generator is None:
        print("[WARNING] Test generator is missing. Evaluation skipped.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] 🔍 Performing Deep Clinical Evaluation...")
    
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title("Clinical Diagnostic Confusion Matrix", fontsize=14)
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('AI Diagnosis (Predicted)')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)

    # 2. ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)

    # 3. Save report to CSV
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_dir, "clinical_metrics_report.csv"))
    
    print(f"[SUCCESS] 📄 Detailed evaluation reports and plots saved in '{save_dir}'")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    # 4. Generate Precision-Recall Curve (Premium Feature)
    plot_precision_recall(y_true, predictions, save_dir)
    
    # 5. Generate Metric Comparison Chart (Premium Feature)
    plot_metric_comparison(report, save_dir)

def plot_precision_recall(y_true, predictions, save_dir):
    """Saves Precision-Recall curve for medical diagnostic validation."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, predictions)
    avg_precision = average_precision_score(y_true, predictions)

    plt.figure(figsize=(8,6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={avg_precision:.2f}')
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300)

def plot_metric_comparison(report, save_dir):
    """Saves a professional bar chart comparing key clinical metrics."""
    metrics = ['accuracy', 'macro avg'] # Focus on these from report dict
    labels = ['Precision', 'Recall', 'F1-Score']
    
    # Extracting values for 'Normal' and 'Pneumonia'
    normal_vals = [report['Normal']['precision'], report['Normal']['recall'], report['Normal']['f1-score']]
    pneumonia_vals = [report['Pneumonia']['precision'], report['Pneumonia']['recall'], report['Pneumonia']['f1-score']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, normal_vals, width, label='Normal', color='#4CAF50')
    ax.bar(x + width/2, pneumonia_vals, width, label='Pneumonia', color='#FF5722')

    ax.set_ylabel('Scores')
    ax.set_title('Cross-Class Metric Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(os.path.join(save_dir, "metric_comparison.png"), dpi=300)
