import argparse
import os
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

# Preprocessing
from utils.dataset import SeaTurtleDataset

# Training and Evaluation
from utils.utils import train_one_epoch_with_accumulation, evaluate_one_epoch, predict_with_knn_sklearn
from modules.models import SwinB_Backbone, ArcFaceLoss

# Federation
# from fedproto.server import FedProtoServerApp
# from fedproto.client import FedProtoClientApp
# from fedproto.task import build_federated_splits

# Logging
from utils.results_writer import save_results, save_training_history

# =========================================================
# Function: Federated Mode
# =========================================================
def with_federation(args: argparse.Namespace, verbose=False):

    # # Configs
    # num_clients = args.num_clients
    # n_way = args.n_way
    # k_shot = args.k_shot
    # n_samples = args.k_shot + args.query
    # episodes = args.episodes
    # eval_episodes = args.eval_episodes
    # test_episodes = args.test_episodes
    # lr = args.lr
    # lambda_align = args.lambda_align
    # lambda_triplet = args.lambda_triplet
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # rounds = args.rounds
    # embedding_dim = args.embedding_dim
    # backbone = args.backbone
    # optimizer = args.optimizer
    # experiment_name = args.experiment_name
    # early_stopping_patience = args.early_stopping_patience

    # # Preprocess data
    # (   
    #     train_dataset, 
    #     val_dataset, 
    #     test_dataset, 
    #     train_sampler, 
    #     val_sampler, 
    #     test_sampler
    # ) = preprocess_data(args)
    # print("\n✓ Dataset and samplers initialized\n")

    # # Re-ID split
    # val_query_dataset, val_gallery_dataset = split_dataset_reid(val_dataset)
    # test_query_dataset, test_gallery_dataset = split_dataset_reid(test_dataset)
    # print("     - ✓ Re-ID datasets ready")

    # args.loss_fn = PrototypicalLoss(n_support=k_shot)
    # print("     - ✓ Global loss function ready")

    # # Split Dataset based on the number of clients
    # client_splits = build_federated_splits(train_dataset, num_clients)
    # print("     - ✓ Client splits ready")

    # # Setup Global Model
    # encoder = None
    # if backbone == "resnet18_aspp":
    #     encoder = ResNet18ASPPEncoder(embedding_dim=embedding_dim)
    # elif backbone == "resnet18":
    #     encoder = ResNet18Encoder(embedding_dim=embedding_dim)
    # else:
    #     raise ValueError(f"Unknown backbone: {backbone}")
    # global_model = encoder.to(device) 

    # global_model_msg = {
    #     'server': 'global',
    #     'model_weights': global_model.state_dict()
    # }
    # clients = []
    # for i in range(num_clients):
    #     train_loader = torch.utils.data.DataLoader(
    #         client_splits[i],
    #         batch_size=64
    #     )
    #     clients.append(FedProtoClientApp(
    #         cid=i,
    #         train_dataset=client_splits[i],
    #         train_loader=train_loader,
    #         n_way=n_way,
    #         k_shot=k_shot,
    #         n_samples=n_samples,
    #         episodes=episodes,
    #         model=backbone,
    #         optimizer=optimizer,
    #         embedding_dim=embedding_dim,
    #         lambda_align=lambda_align,
    #         lambda_triplet=lambda_triplet,
    #         lr=lr,
    #     ))
    #     clients[i].set_model_weights(global_model_msg)

    # print("     - ✓ Clients ready")

    # server = FedProtoServerApp(backbone=encoder, device=device)
    # print("     - ✓ Server ready")

    # # Results directory
    # results_dir = Path("results") / experiment_name
    # results_dir.mkdir(parents=True, exist_ok=True)

    # history = {
    #     'train_proto_loss': [],
    #     'train_triplet_loss': [],
    #     'train_acc': [],
    #     'val_loss': [],
    #     'val_acc': [],
    #     'val_mAP': [],
    #     'val_rank1': [],
    #     'val_rank5': []
    # }


    # # Early Stopping Configs
    # early_stopping_counter = 0
    # best_acc = 0.0

    # for round in range(rounds):
    #     if early_stopping_counter > early_stopping_patience:
    #         print("Early stopping triggered. Ending training.")
    #         break

    #     print(f"\n********** ROUND {round + 1}/{rounds} **********")

    #     round_losses = []
    #     round_accs = []

    #     # Run local episodic loop for n number of episodes
    #     for client in clients:
    #         client_msg = client.fit()
        
    #     # ... Insert New Training code

    #     mAP, rank1, rank5 = eval_reid_fn(
    #         model=server.global_encoder,
    #         query_dataset=val_query_dataset,
    #         gallery_dataset=val_gallery_dataset,
    #         device=device
    #     )

    #     history['val_mAP'].append(mAP)
    #     history['val_rank1'].append(rank1)
    #     history['val_rank5'].append(rank5)

    #     if mAP > best_acc:
    #         best_acc = mAP
    #         model_path = results_dir / f"best_global_model.pth"
    #         torch.save(server.global_encoder.state_dict(), model_path)
    #         print(f"     - ✓ New best model saved at round {round+1} with VAL acc: {best_acc*100:.2f}%")
    #         early_stopping_counter = 0
    #     else:
    #         early_stopping_counter += 1
    
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    # save_training_history(history, results_dir) # Save Training History
    # mAP, rank1, rank5 = eval_reid_fn(
    #     model=server.global_encoder,
    #     query_dataset=test_query_dataset,
    #     gallery_dataset=test_gallery_dataset,
    #     device=device
    # )
    # print(f"     - ✓ TEST Re-ID mAP: {mAP*100:.2f}%, Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%")
    # save_results(results_dir, loss, acc) # Save Test Results

    return


# Function: Non-Federated Mode
# Function: Non-Federated Mode
def without_federation(args: argparse.Namespace, verbose=False):
    # Dataset Configs
    data_dir = args.data_dir
    split_mode = args.split_mode
    segment = args.segment

    # Gradient Accumulation Configs
    actual_batch_size = 32 
    effective_batch_size = 128 
    accumulation_steps = effective_batch_size // actual_batch_size 
    
    print(f"Using gradient accumulation:")
    print(f"  Actual batch size: {actual_batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")

    # Preprocess Data
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = preprocess_data(
        data_dir, split_mode, segment, batch_size=actual_batch_size, verbose=verbose
    )

    # Model Configs
    epochs = args.epochs
    lr = args.lr
    optimizer_name = args.optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience = args.early_stopping_patience
    experiment_name = args.experiment_name

    # Get number of classes
    num_classes = train_dataset.get_num_classes()
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Batch size (actual/effective): {actual_batch_size}/{effective_batch_size}")
    print(f"{'='*60}\n")

    # Model Initialization
    embedding_dim = args.embedding_dim
    model = SwinB_Backbone(embedding_dim=embedding_dim).to(device)
    
    loss_fn = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        s=64.0,  # As per paper
        m=0.5    # As per paper
    ).to(device)

    # Optimizer setup
    if optimizer_name == "adam":
        optimizer = Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=lr
        )
    else:
        optimizer = SGD(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=lr,
            momentum=0.9
        )
    
    # Cosine annealing scheduler (as per paper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Results directory
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Early stopping setup
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    best_model_path = results_dir / "best_model.pth"
    last_model_path = results_dir / "last_model.pth"

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING WITH GRADIENT ACCUMULATION")
    print("="*60 + "\n")

    for epoch in range(epochs):
        print(f"{'='*60}")
        print(f"EPOCH [{epoch+1}/{epochs}]")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch_with_accumulation(
            model, loss_fn, optimizer, train_loader, device, 
            epoch, epochs, accumulation_steps
        )
        
        # Validate
        val_loss, val_acc = evaluate_one_epoch(model, loss_fn, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save last model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, last_model_path)
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            print(f"  ✓ Saved to: {best_model_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            print(f"  Best Val Acc so far: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"Best model was at epoch {best_epoch+1} with Val Acc: {best_val_acc:.2f}%")
            break
        
        print()
    
    # Save training history
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
    
    save_training_history(history, results_dir)
    print(f"✓ Training history saved")
    
    # Load best model for evaluation
    print(f"\n{'='*60}")
    print("LOADING BEST MODEL FOR EVALUATION")
    print(f"{'='*60}")
    
    if best_model_path.exists():
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        print(f"✓ Model loaded successfully!")
        print(f"  - Epoch: {checkpoint['epoch']+1}")
        print(f"  - Train Acc: {checkpoint['train_acc']:.2f}%")
        print(f"  - Val Acc: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"⚠ Warning: Best model not found at {best_model_path}")
        print(f"Using last trained model instead")
    
    # Test with different k values
    print("\n" + "="*60)
    print("TESTING WITH DIFFERENT K VALUES")
    print("="*60)
    
    k_values = [1, 3, 5, 10]
    test_results = {}
    
    for k in k_values:
        print(f"\n--- Testing with k={k} ---")
        accuracy, preds, labels = predict_with_knn_sklearn(
            model, 
            train_loader, 
            test_loader, 
            device, 
            k=k,
            verbose=True
        )
        test_results[f'test_acc_k{k}'] = accuracy
    
    # Save test results
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)
    
    results_file = results_dir / "test_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch+1})\n")
        f.write(f"Total Epochs Trained: {len(history['train_loss'])}\n\n")
        f.write("Test Accuracies (k-NN):\n")
        f.write("-" * 30 + "\n")
        for k_name, acc in test_results.items():
            f.write(f"{k_name}: {acc:.2f}%\n")
    
    print(f"✓ Test results saved to: {results_file}")
    
    # Save model info
    model_info_file = results_dir / "model_info.txt"
    with open(model_info_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Backbone: Swin-B\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Num Classes: {num_classes}\n")
        f.write(f"Loss: ArcFace (s=64.0, m=0.5)\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {actual_batch_size} (effective: {effective_batch_size})\n")
        f.write(f"Accumulation Steps: {accumulation_steps}\n")
        f.write(f"\nBest Model Path: {best_model_path}\n")
        f.write(f"Last Model Path: {last_model_path}\n")
    
    print(f"✓ Model info saved to: {model_info_file}")
    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {results_dir}")
    print(f"{'='*60}\n")
    
    return model, loss_fn, history, test_results
    

# Function: Preprocessing Pipeline
def preprocess_data(data_dir, split_mode, segment, batch_size=128, verbose=False):
    
    # Segment selection
    segment = segment.lower().strip()
    valid_segments = ["full_img", "turtle", "flipper", "head"]
    assert segment in valid_segments, f"Invalid segment: {segment}. Must be one of {valid_segments}."

    # Load Metadata
    segment_file_map = {
        "full_img": "metadata_splits.csv",
        "turtle": "metadata_splits_turtle.csv",
        "flipper": "metadata_splits_flipper.csv",
        "head": "metadata_splits_head.csv",
    }
    metadata_path = Path(data_dir) / segment_file_map[segment]
    assert metadata_path.exists(), f"Metadata file not found: {metadata_path}"
    metadata_df = pd.read_csv(metadata_path)

    # Split Mode Selection
    split_column = {
        "closed": "split_closed",
        "random": "split_closed_random",
        "open": "split_open"
    }[split_mode]
    train_df = metadata_df[metadata_df[split_column] == "train"].copy()
    val_df   = metadata_df[metadata_df[split_column] == "valid"].copy()
    test_df  = metadata_df[metadata_df[split_column] == "test"].copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"DATASET INFORMATION")
        print(f"{'='*60}")
        print(f"Segment: {segment}")
        print(f"Split mode: {split_mode}")
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"{'='*60}\n")

    # Building Datasets
    train_dataset = SeaTurtleDataset(train_df, data_dir, train=True, verbose=verbose)
    val_dataset   = SeaTurtleDataset(val_df, data_dir, train=False, verbose=verbose)
    test_dataset  = SeaTurtleDataset(test_df, data_dir, train=False, verbose=verbose)

    # Building DataLoaders
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


    





    
