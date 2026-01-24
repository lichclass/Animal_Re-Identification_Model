# results_writer.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def save_results(results_dir, test_loss, test_acc, test_mAP=None, test_rank1=None, test_rank5=None):
    """Save final test results"""
    out = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }
    
    # Add Re-ID metrics if provided
    if test_mAP is not None:
        out["test_mAP"] = float(test_mAP)
    if test_rank1 is not None:
        out["test_rank1"] = float(test_rank1)
    if test_rank5 is not None:
        out["test_rank5"] = float(test_rank5)
    
    results_path = os.path.join(results_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=4)
    
    print(f"\n✓ Test results saved to {results_path}")


def save_training_history(history, results_dir):
    """
    Save training history with plots.
    
    Args:
        history: Dict with keys for standard training:
            - 'train_loss': list of training losses per epoch
            - 'train_acc': list of training accuracies per epoch
            - 'val_loss': list of validation losses per epoch
            - 'val_acc': list of validation accuracies per epoch
            
            OR for federated/Re-ID training:
            - 'train_proto_loss': list of proto losses per round
            - 'train_triplet_loss': list of triplet losses per round
            - 'train_acc': list of accuracies per round
            - 'val_loss': list of validation losses per round
            - 'val_acc': list of validation accuracies per round
            - 'val_mAP': list of validation mAPs per round
            - 'val_rank1': list of validation rank-1 per round
            - 'val_rank5': list of validation rank-5 per round
        
        results_dir: Directory to save results
    """
    results_dir = str(results_dir)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save history as JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved to {history_path}")
    
    # Determine if this is epoch-based or round-based training
    num_iterations = len(history.get('train_acc', []))
    if num_iterations == 0:
        print("⚠ Warning: Empty training history, skipping plots")
        return
    
    iterations = range(1, num_iterations + 1)
    is_federated = 'train_proto_loss' in history
    x_label = 'Round' if is_federated else 'Epoch'
    
    # Plot 1: Loss curves
    plt.figure(figsize=(12, 6))
    
    if is_federated and 'train_proto_loss' in history and 'train_triplet_loss' in history:
        # Federated learning losses
        plt.plot(iterations, history['train_proto_loss'], 'b-o', label='Proto Loss', linewidth=2, markersize=4)
        plt.plot(iterations, history['train_triplet_loss'], 'r-s', label='Triplet Loss', linewidth=2, markersize=4)
        title = 'Training Losses (Federated)'
    else:
        # Standard training losses
        if 'train_loss' in history:
            plt.plot(iterations, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        if 'val_loss' in history:
            plt.plot(iterations, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
        title = 'Training and Validation Loss'
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=300)
    plt.close()
    print(f"✓ Loss curves plot saved")
    
    # Plot 2: Accuracy Curves
    plt.figure(figsize=(12, 6))
    
    if 'train_acc' in history:
        plt.plot(iterations, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    if 'val_acc' in history:
        plt.plot(iterations, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curves.png"), dpi=300)
    plt.close()
    print(f"✓ Accuracy curves plot saved")
    
    # Plot 3: Re-ID Metrics (only if available)
    if 'val_mAP' in history and 'val_rank1' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, history['val_mAP'], 'g-o', label='mAP', linewidth=2, markersize=4)
        plt.plot(iterations, history['val_rank1'], 'b-s', label='Rank-1', linewidth=2, markersize=4)
        if 'val_rank5' in history:
            plt.plot(iterations, history['val_rank5'], 'r-^', label='Rank-5', linewidth=2, markersize=4)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Re-ID Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "reid_metrics.png"), dpi=300)
        plt.close()
        print(f"✓ Re-ID metrics plot saved")
    
    # Plot 4: Summary statistics (text file)
    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total {x_label}s: {num_iterations}\n\n")
        
        if 'train_loss' in history:
            f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Best Train Loss:  {min(history['train_loss']):.4f}\n")
        
        if 'val_loss' in history:
            f.write(f"Final Val Loss:   {history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Val Loss:    {min(history['val_loss']):.4f}\n")
        
        f.write("\n")
        
        if 'train_acc' in history:
            f.write(f"Final Train Acc:  {history['train_acc'][-1]:.2f}%\n")
            f.write(f"Best Train Acc:   {max(history['train_acc']):.2f}%\n")
        
        if 'val_acc' in history:
            f.write(f"Final Val Acc:    {history['val_acc'][-1]:.2f}%\n")
            f.write(f"Best Val Acc:     {max(history['val_acc']):.2f}%\n")
            best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
            f.write(f"Best Val Acc at {x_label}: {best_epoch}\n")
        
        if 'val_mAP' in history:
            f.write("\n")
            f.write(f"Final Val mAP:    {history['val_mAP'][-1]:.2f}%\n")
            f.write(f"Best Val mAP:     {max(history['val_mAP']):.2f}%\n")
        
        if 'val_rank1' in history:
            f.write(f"Final Val Rank-1: {history['val_rank1'][-1]:.2f}%\n")
            f.write(f"Best Val Rank-1:  {max(history['val_rank1']):.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Training summary saved to {summary_path}")


def save_metrics(results_path, map1, map5):
    """Legacy function for backward compatibility"""
    out = {
        "mAP@1": map1,
        "mAP@5": map5
    }
    with open(os.path.join(results_path, "metrics.json"), "w") as f:
        json.dump(out, f, indent=4)