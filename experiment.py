import argparse
import os
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader

from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss

# Preprocessing
from utils.dataset import SeaTurtleDataset

# Training and Evaluation
from utils.utils import train_one_epoch, evaluate_one_epoch
from modules.models import SwinB_Backbone

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
def without_federation(args: argparse.Namespace, verbose=False):
    
    # Dataset Configs
    data_dir = args.data_dir
    split_mode = args.split_mode
    segment = args.segment
    verbose = args.verbose

    # Preprocess Data
    _, _, _, train_loader, val_loader, test_loader =  preprocess_data(data_dir, split_mode, segment, verbose)

    # Model Configs
    epochs = args.epochs
    lr = args.lr
    optimizer_name = args.optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience = args.early_stopping_patience

    model = SwinB_Backbone().to(device)
    optim = (Adam(model.parameters(), lr=lr) if optimizer_name == "adam"
        else SGD(model.parameters(), lr=lr, momentum=0.9))
    loss_fn = CrossEntropyLoss()

    best_val_acc = 0.0
    early_stopping = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, 
            train_loader, 
            optim, 
            loss_fn, 
            description=f"Epoch {epoch+1}/{epochs} - Training"
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, 
            val_loader, 
            loss_fn, 
            description=f"Epoch {epoch+1}/{epochs} - Validation"
        )   

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_accs)

        change = val_acc - best_val_acc
        if change < args.early_threshold:
            early_stopping += 1
        else:
            early_stopping = 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - New best validation accuracy: {best_val_acc*100:.2f}%")
        
        if early_stopping > early_stopping_patience:
            print("Stopping due to lack of improvement...")

    test_loss, test_acc = evaluate_one_epoch(
        model, 
        test_loader, 
        loss_fn, 
        description=f"Testing"
    )

    print(f"Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")
        

# Function: Preprocessing Pipeline
def preprocess_data(data_dir, split_mode, segment, verbose=False):
    
    # Segment selection
    segment = segment.lower().strip()
    valid_segments = ["full_img","turtle", "flipper", "head"]
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

    # Building Datasets
    train_dataset = SeaTurtleDataset(train_df, data_dir, verbose=False)
    val_dataset   = SeaTurtleDataset(val_df, data_dir, verbose=False)
    test_dataset = SeaTurtleDataset(test_df, data_dir, verbose=False)

    # Building DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


    





    
