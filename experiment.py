import argparse
import os
import torch
import pandas as pd
import numpy as np
import random

# Preprocessing
from dataset import SeaTurtleDataset
from tasksampler import FewShotTaskSampler

# ProtoNet components
from modules.resnet_aspp import ResNet18ASPPEncoder
from modules.resnet18 import ResNet18Encoder
from modules.protonetloss import PrototypicalLoss

# Training / Evaluation
from trainer import train_one_epoch
from evaluator import evaluate_one_epoch

# Federation
from fedproto.server import FedProtoServerApp
from fedproto.client import FedProtoClientApp
from fedproto.task import build_federated_splits

# Parallelization
from concurrent.futures import ThreadPoolExecutor


# =========================================================
# Function: Federated Mode
# =========================================================
def with_federation(args: argparse.Namespace, verbose=False):
    # =========================================================
    # 1. PREPROCESS DATA
    # =========================================================
    (   
        train_dataset, 
        val_dataset, 
        test_dataset, 
        train_sampler, 
        val_sampler, 
        test_sampler
    ) = preprocess_data(args)
    print("\n✓ Dataset and samplers initialized\n")

    # =========================================================
    # 2. GLOBAL LOSS FUNCTION
    # =========================================================
    args.loss_fn = PrototypicalLoss(n_support=args.k_shot)
    print("     - ✓ Global loss function ready")

    # Configs
    num_clients = args.num_clients
    n_way = args.n_way
    k_shot = args.k_shot
    n_samples = args.k_shot + args.query
    episodes = args.episodes
    eval_episodes = args.eval_episodes
    test_episodes = args.test_episodes
    lr = args.lr
    lambda_align = args.lambda_align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rounds = args.rounds
    embedding_dim = args.embedding_dim
    backbone = args.backbone
    optimizer = args.optimizer

    # Split Dataset based on the number of clients
    client_splits = build_federated_splits(train_dataset, num_clients)
    print("     - ✓ Client splits ready")

    # Setup Global Model
    encoder = None
    if backbone == "resnet18_aspp":
        encoder = ResNet18ASPPEncoder(embedding_dim=embedding_dim)
    elif backbone == "resnet18":
        encoder = ResNet18Encoder(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    global_model = encoder.to(device) # Initialize global model for unified weights

    # Create Clients based on the number of clients
    clients = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(
            client_splits[i],
            batch_size=64
        )
        clients.append(FedProtoClientApp(
            cid=i,
            train_dataset=client_splits[i],
            train_loader=train_loader,
            n_way=n_way,
            k_shot=k_shot,
            n_samples=n_samples,
            episodes=episodes,
            model=backbone,
            optimizer=optimizer,
            embedding_dim=embedding_dim,
            lr=lr,
        ))
        clients[i].set_model_weights(global_model.state_dict())

    print("     - ✓ Clients ready")

    # Setup Server
    server = FedProtoServerApp(backbone=encoder, lr=lr)
    print("     - ✓ Server ready")

    # Start Communications Rounds
    for round in range(rounds):
        print(f"\n********** ROUND {round + 1}/{rounds} **********")

        # Run local episodic loop for n number of episodes
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            results = list(executor.map(lambda c: c.fit(), clients))

        for cid, (loss, acc) in enumerate(results):
            print(f"(Client {cid}) Train Loss: {loss:.4f} | Train Acc: {acc*100:0.2f}%")

        # Get the prototypes from the clients
        client_protos = {}
        for client in clients:
            client_protos[client.cid] = client.get_local_prototypes()
            print(f"     - ✓ Client {client.cid} prototypes ready")

        # Sends the prototypes to the server and aggregate
        global_prototypes = server.aggregate(client_protos)
        print("     - ✓ Global prototypes ready")

        # Sends the global prototypes to the clients
        for client in clients:
            client.set_global_prototypes(global_prototypes)
        print('     - ✓ Global prototypes sent to clients')

        chosen_client = random.choice(clients)
        print(f"Evaluating Client {chosen_client.cid} Model...")
        model_weights = chosen_client.get_model_weights()
        server.set_global_model_weights(model_weights)
        server.evaluate(val_dataset)

        # Update weights
        for client in clients:
            client.set_model_weights(server.get_global_model_weights())
            print(f"     - ✓ Client {client.cid} weights updated")


# =========================================================
# Function: Non-Federated Mode
# =========================================================
def without_federation(args: argparse.Namespace, verbose=False):
    pass

    # Preprocess data
    (
        train_dataset, 
        val_dataset, 
        test_dataset, 
        train_sampler, 
        val_sampler, 
        test_sampler
    ) = preprocess_data(args)

    if verbose:
        print("\n✓ Dataset and samplers initialized\n")

    # =========================================================
    # BUILD MODEL
    # =========================================================
    
    if verbose:
        print("===============================")
        print("BUILDING THE MODEL")
        print("===============================\n")

    if args.backbone == "resnet18":
        encoder = ResNet18Encoder(embedding_dim=args.embedding_dim)
    else:
        encoder = ResNet18ASPPEncoder(embedding_dim=args.embedding_dim)

    model = encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    # Sampler Configs
    n_way = args.n_way
    k_shot = args.k_shot
    n_samples = args.k_shot + args.query

    # ProtoNet loss
    loss_fn = PrototypicalLoss(n_support=k_shot)

    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    )

    if verbose:
        print("     - ✓ Model initialized")
        print("     - ✓ Loss function initialized")
        print("     - ✓ optimizer initialized")
    
    # =========================================================
    # TRAINING LOOP
    # =========================================================
    if verbose:
        print("\n===============================")
        print("TRAINING THE MODEL")
        print("===============================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(1, args.epochs + 1):
        print(f"\n********** EPOCH {epoch}/{args.epochs} **********")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_dataset=train_dataset,
            task_sampler=train_sampler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        # ================================
        # Validation (mAP@1, mAP@5)
        # ================================
        map1, map5 = evaluate_one_epoch(
            model=model,
            eval_dataset=val_dataset,
            task_sampler=val_sampler,
            n_support=k_shot,
            device=device
        )

        print(f"VAL mAP@1: {map1*100:.2f}% | VAL mAP@5: {map5*100:.2f}%")

    print("\nTraining Complete!\n")

    # ================================
    # Testing (mAP@1, mAP@5)
    # ================================
    if verbose:
        print("\n===============================")
        print("TESTING THE MODEL")
        print("===============================\n")

    map1, map5 = evaluate_one_epoch(
        model=model,
        eval_dataset=test_dataset,
        task_sampler=test_sampler,
        n_support=k_shot,
        device=device
    )

    print(f"TEST mAP@1: {map1*100:.2f}% | TEST mAP@5: {map5*100:.2f}%")


# =========================================================
# Function: Preprocessing Pipeline
# =========================================================
def preprocess_data(args: argparse.Namespace, verbose=False):

    if verbose:
        print("===============================")
        print("PREPROCESSING DATA")
        print("===============================")

    # ------------------------------
    # Pipeline 1: Segment selection
    # 
    # Description: 
    #    Select the segment to train on. 
    #    Segment = "turtle" (full body), "flipper", "head"
    # ------------------------------

    if verbose:
        print("\nStage 1: Segment Selection\n")

    segment = args.segment.lower().strip()
    valid_segments = ["turtle", "flipper", "head"]

    # Check if config segment is valid
    if segment not in valid_segments:
        raise ValueError(f"--segment must be one of {valid_segments}")

    segment_file_map = {
        "turtle": "metadata_splits_turtle.csv",
        "flipper": "metadata_splits_flipper.csv",
        "head": "metadata_splits_head.csv"
    }

    metadata_path = os.path.join(args.data_dir, segment_file_map[segment])
    metadata_df = pd.read_csv(metadata_path)

    if verbose:
        print(f"    - ✓ Selected segment: {segment}")
        print(f"    - ✓ Loaded {len(metadata_df)} entries")

    # ------------------------------
    # Pipeline 2: Split Mode Selection
    # 
    # Description: 
    #    Select the split mode to use
    #    Splits = "closed", "random", "open"
    # ------------------------------

    if verbose:
        print("\nStage 2: Split Mode Selection\n")

    split_column = {
        "closed": "split_closed",
        "random": "split_closed_random",
        "open": "split_open"
    }[args.split_mode]

    if verbose:
        print(f"    - ✓ Using split mode: {args.split_mode}")
        print(f"    - ✓ Column: {split_column}")

    train_df = metadata_df[metadata_df[split_column] == "train"].copy()
    val_df   = metadata_df[metadata_df[split_column] == "valid"].copy()
    test_df  = metadata_df[metadata_df[split_column] == "test"].copy()

    if verbose:
        print(f"    - ✓ Train samples: {len(train_df)}")
        print(f"    - ✓ Val samples:   {len(val_df)}")
        print(f"    - ✓ Test samples:  {len(test_df)}")

    # ------------------------------
    # Pipeline 3: Building Dataset Loaders
    # 
    # Description: 
    #    Build dataset loaders
    # ------------------------------

    if verbose:
        print("\nStage 3: Building Dataset Loaders\n")

    train_dataset = SeaTurtleDataset(train_df, args.data_dir, verbose=False)
    val_dataset   = SeaTurtleDataset(val_df, args.data_dir, verbose=False)
    test_dataset = SeaTurtleDataset(test_df, args.data_dir, verbose=False)

    print(f"    - ✓ Train dataset ready: {len(train_dataset)} samples")
    print(f"    - ✓ Val dataset ready:   {len(val_dataset)} samples")
    print(f"    - ✓ Test dataset ready: {len(test_dataset)} samples")

    # ------------------------------
    # Pipeline 4: Building Task Samplers
    # 
    # Description: 
    #    Build task samplers
    # ------------------------------

    if verbose:
        print("\nStage 4: Building Task Samplers\n")

    # Use dataset-level mapping
    train_labels = train_dataset.df["identity"].map(train_dataset.identity_to_idx).values
    val_labels   = val_dataset.df["identity"].map(val_dataset.identity_to_idx).values
    test_labels  = test_dataset.df["identity"].map(test_dataset.identity_to_idx).values

    # Sampler Configs
    n_way = args.n_way
    k_shot = args.k_shot
    n_samples = args.k_shot + args.query

    train_sampler = FewShotTaskSampler(
        labels=train_labels,
        n_way=n_way,
        n_samples=n_samples,
        iterations=args.episodes,
        allow_replacement=False
    )

    val_sampler = FewShotTaskSampler(
        labels=val_labels,
        n_way=n_way,
        n_samples=n_samples,
        iterations=args.eval_episodes,
        allow_replacement=False
    )

    test_sampler = FewShotTaskSampler(
        labels=test_labels,
        n_way=n_way,
        n_samples=n_samples,
        iterations=args.test_episodes,
        allow_replacement=False
    )

    print("     - ✓ Train sampler ready")
    print("     - ✓ Val sampler ready")
    print("     - ✓ Test sampler ready")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_sampler, 
        val_sampler, 
        test_sampler,
    )


    





    
