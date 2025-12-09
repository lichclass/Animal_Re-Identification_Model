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


def save_training_history(results_dir, history):
    """
    Save training history with plots.
    
    Args:
        results_dir: Directory to save results
        history: Dict with keys:
            - 'train_proto_loss': list of proto losses per round
            - 'train_triplet_loss': list of triplet losses per round
            - 'train_acc': list of accuracies per round
            - 'val_loss': list of validation losses per round
            - 'val_acc': list of validation accuracies per round
            - 'val_mAP': list of validation mAPs per round
            - 'val_rank1': list of validation rank-1 per round
    """
    results_dir = str(results_dir)
    
    # Save history as JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved to {history_path}")
    
    rounds = range(1, len(history['train_acc']) + 1)
    
    # Plot 1: Training Losses
    if 'train_proto_loss' in history and 'train_triplet_loss' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, history['train_proto_loss'], 'b-o', label='Proto Loss', linewidth=2)
        plt.plot(rounds, history['train_triplet_loss'], 'r-s', label='Triplet Loss', linewidth=2)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Losses', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "training_losses.png"), dpi=300)
        plt.close()
        print(f"✓ Training losses plot saved")
    
    # Plot 2: Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        plt.plot(rounds, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curves.png"), dpi=300)
    plt.close()
    print(f"✓ Accuracy curves saved")
    
    # Plot 3: Re-ID Metrics
    if 'val_mAP' in history and 'val_rank1' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, history['val_mAP'], 'g-o', label='mAP', linewidth=2)
        plt.plot(rounds, history['val_rank1'], 'b-s', label='Rank-1', linewidth=2)
        if 'val_rank5' in history:
            plt.plot(rounds, history['val_rank5'], 'r-^', label='Rank-5', linewidth=2)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Re-ID Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "reid_metrics.png"), dpi=300)
        plt.close()
        print(f"✓ Re-ID metrics plot saved")


def save_metrics(results_path, map1, map5):
    """Legacy function for backward compatibility"""
    out = {
        "mAP@1": map1,
        "mAP@5": map5
    }
    with open(os.path.join(results_path, "metrics.json"), "w") as f:
        json.dump(out, f, indent=4)