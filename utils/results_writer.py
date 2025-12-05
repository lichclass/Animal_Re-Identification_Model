# results_writer.py
import os
import json
import matplotlib.pyplot as plt

def save_metrics(results_path, map1, map5):
    out = {
        "mAP@1": map1,
        "mAP@5": map5
    }
    with open(os.path.join(results_path, "metrics.json"), "w") as f:
        json.dump(out, f, indent=4)


def save_training_plot(results_path, losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "loss_curve.png"))
    plt.close()


def save_map_plot(results_path, map1_list, map5_list):
    plt.figure(figsize=(6,4))
    plt.plot(map1_list, label="mAP@1")
    plt.plot(map5_list, label="mAP@5")
    plt.legend()
    plt.title("Validation mAP Curves")
    plt.xlabel("Evaluation Step")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "map_curve.png"))
    plt.close()

def save_results(results_path, test_loss, test_acc):
    out = {
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
    with open(os.path.join(results_path, "test_results.json"), "w") as f:
        json.dump(out, f, indent=4)