import argparse
import os
import torch
import pandas as pd
import numpy as np

from pathlib import Path

# Preprocessing
from utils.dataset import SeaTurtleDataset
from utils.tasksampler import FewShotTaskSampler
from utils.utils import split_dataset_reid

# ProtoNet components
from modules.resnet_aspp import ResNet18ASPPEncoder
from modules.resnet18 import ResNet18Encoder
from modules.protonetloss import PrototypicalLoss

# Training / Evaluation
from utils.evaluator import evaluate_one as eval_fn
from utils.evaluator import evaluate_reid as eval_reid_fn

# Federation
from fedproto.server import FedProtoServerApp
from fedproto.client import FedProtoClientApp
from fedproto.task import build_federated_splits

# Logging
from utils.results_writer import save_results, save_training_history

# =========================================================
# Function: Federated Mode
# =========================================================
def with_federation(args: argparse.Namespace, verbose=False):

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
    lambda_triplet = args.lambda_triplet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rounds = args.rounds
    embedding_dim = args.embedding_dim
    backbone = args.backbone
    optimizer = args.optimizer
    experiment_name = args.experiment_name
    early_stopping_patience = args.early_stopping_patience

    # Preprocess data
    (   
        train_dataset, 
        val_dataset, 
        test_dataset, 
        train_sampler, 
        val_sampler, 
        test_sampler
    ) = preprocess_data(args)
    print("\n✓ Dataset and samplers initialized\n")

    # Re-ID split
    val_query_dataset, val_gallery_dataset = split_dataset_reid(val_dataset)
    test_query_dataset, test_gallery_dataset = split_dataset_reid(test_dataset)
    print("     - ✓ Re-ID datasets ready")

    args.loss_fn = PrototypicalLoss(n_support=k_shot)
    print("     - ✓ Global loss function ready")

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
    global_model = encoder.to(device) 

    global_model_msg = {
        'server': 'global',
        'model_weights': global_model.state_dict()
    }
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
            lambda_align=lambda_align,
            lambda_triplet=lambda_triplet,
            lr=lr,
        ))
        clients[i].set_model_weights(global_model_msg)

    print("     - ✓ Clients ready")

    server = FedProtoServerApp(backbone=encoder, device=device)
    print("     - ✓ Server ready")

    # Results directory
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_proto_loss': [],
        'train_triplet_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mAP': [],
        'val_rank1': [],
        'val_rank5': []
    }


    # Early Stopping Configs
    early_stopping_counter = 0
    best_acc = 0.0
    best_mAP = 0.0

    for round in range(rounds):
        if early_stopping_counter > early_stopping_patience:
            print("Early stopping triggered. Ending training.")
            break

        print(f"\n********** ROUND {round + 1}/{rounds} **********")

        round_proto_losses = []
        round_triplet_losses = []
        round_accs = []

        # Run local episodic loop for n number of episodes
        for client in clients:
            client_msg = client.fit()
            round_proto_losses.append(client_msg['proto_loss'])
            round_triplet_losses.append(client_msg['triplet_loss'])
            round_accs.append(client_msg['acc'])
            print(f"     - ✓ Client {client_msg['client']} training complete | Proto Loss: {client_msg['proto_loss']:.4f} | Triplet Loss: {client_msg['triplet_loss']:.4f} | Acc: {client_msg['acc']*100:.2f}%")
        
        avg_proto_loss = np.mean(round_proto_losses)
        avg_triplet_loss = np.mean(round_triplet_losses)
        avg_acc = np.mean(round_accs)

        history['train_proto_loss'].append(avg_proto_loss)
        history['train_triplet_loss'].append(avg_triplet_loss)
        history['train_acc'].append(avg_acc)

        # Get the prototypes from the clients
        client_protos = {}
        for client in clients:
            client_msg = client.get_local_prototypes()
            client_protos[client_msg['client']] = {
                'prototypes': client_msg['prototypes'],
                'counts': client_msg['counts']
            }
            print(f"     - ✓ Client {client_msg['client']} prototypes ready")

        # Sends the prototypes to the server and aggregate
        global_prototypes = server.aggregate(client_protos)
        print("     - ✓ Global prototypes ready")

        # Sends the global prototypes to the clients
        for client in clients:
            client.set_global_prototypes(global_prototypes)
        print('     - ✓ Global prototypes sent to clients')

        server.fedAvg(clients)
        print("     - ✓ Global model weights updated and sent to clients")

        print(f"(Validation) Evaluating the Global Client...")
        loss, acc = eval_fn(
            model=server.global_encoder,
            eval_dataset=val_dataset,
            task_sampler=val_sampler,
            n_support=k_shot,
            device=device
        )
        print(f"     - ✓ VAL Loss: {loss:.4f} | VAL Acc: {acc*100:.2f}%")

        history['val_loss'].append(loss)
        history['val_acc'].append(acc)

        mAP, rank1, rank5 = eval_reid_fn(
            model=server.global_encoder,
            query_dataset=val_query_dataset,
            gallery_dataset=val_gallery_dataset,
            device=device
        )
        print(f"     - ✓ VAL Re-ID mAP: {mAP*100:.2f}%, Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%")

        history['val_mAP'].append(mAP)
        history['val_rank1'].append(rank1)
        history['val_rank5'].append(rank5)

        if mAP > best_mAP:
            best_mAP = mAP
            best_acc = acc
            model_path = results_dir / f"best_global_model.pth"
            torch.save(server.global_encoder.state_dict(), model_path)
            print(f"     - ✓ New best model saved at round {round+1} with VAL mAP: {best_mAP*100:.2f}%")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_training_history(history, results_dir)

    print(f"(Test) Evaluating the Global Client...")
    loss, acc = eval_fn(
        model=server.global_encoder,
        eval_dataset=test_dataset,
        task_sampler=test_sampler,
        n_support=k_shot,
        device=device
    )
    print(f"     - ✓ TEST Loss: {loss:.4f} | TEST Acc: {acc*100:.2f}%")
    mAP, rank1, rank5 = eval_reid_fn(
        model=server.global_encoder,
        query_dataset=test_query_dataset,
        gallery_dataset=test_gallery_dataset,
        device=device
    )
    print(f"     - ✓ TEST Re-ID mAP: {mAP*100:.2f}%, Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%")


    save_results(results_dir, loss, acc)

    return

# =========================================================
# Function: Non-Federated Mode
# =========================================================
def without_federation(args: argparse.Namespace, verbose=False):
    pass

    # # Preprocess data
    # (
    #     train_dataset, 
    #     val_dataset, 
    #     test_dataset, 
    #     train_sampler, 
    #     val_sampler, 
    #     test_sampler
    # ) = preprocess_data(args)

    # if verbose:
    #     print("\n✓ Dataset and samplers initialized\n")

    # # =========================================================
    # # BUILD MODEL
    # # =========================================================
    
    # if verbose:
    #     print("===============================")
    #     print("BUILDING THE MODEL")
    #     print("===============================\n")

    # if args.backbone == "resnet18":
    #     encoder = ResNet18Encoder(embedding_dim=args.embedding_dim)
    # else:
    #     encoder = ResNet18ASPPEncoder(embedding_dim=args.embedding_dim)

    # model = encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    # # Sampler Configs
    # n_way = args.n_way
    # k_shot = args.k_shot
    # n_samples = args.k_shot + args.query

    # # ProtoNet loss
    # loss_fn = PrototypicalLoss(n_support=k_shot)

    # optimizer = (
    #     torch.optim.Adam(model.parameters(), lr=args.lr)
    #     if args.optimizer == "adam"
    #     else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # )

    # if verbose:
    #     print("     - ✓ Model initialized")
    #     print("     - ✓ Loss function initialized")
    #     print("     - ✓ optimizer initialized")
    
    # # =========================================================
    # # TRAINING LOOP
    # # =========================================================
    # if verbose:
    #     print("\n===============================")
    #     print("TRAINING THE MODEL")
    #     print("===============================")
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # for epoch in range(1, args.epochs + 1):
    #     print(f"\n********** EPOCH {epoch}/{args.epochs} **********")

    #     train_loss, train_acc = train_one_epoch(
    #         model=model,
    #         train_dataset=train_dataset,
    #         task_sampler=train_sampler,
    #         loss_fn=loss_fn,
    #         optimizer=optimizer,
    #         device=device
    #     )

    #     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

    #     # ================================
    #     # Validation (mAP@1, mAP@5)
    #     # ================================
    #     map1, map5 = evaluate_one_epoch(
    #         model=model,
    #         eval_dataset=val_dataset,
    #         task_sampler=val_sampler,
    #         n_support=k_shot,
    #         device=device
    #     )

    #     print(f"VAL mAP@1: {map1*100:.2f}% | VAL mAP@5: {map5*100:.2f}%")

    # print("\nTraining Complete!\n")

    # # ================================
    # # Testing (mAP@1, mAP@5)
    # # ================================
    # if verbose:
    #     print("\n===============================")
    #     print("TESTING THE MODEL")
    #     print("===============================\n")

    # map1, map5 = evaluate_one_epoch(
    #     model=model,
    #     eval_dataset=test_dataset,
    #     task_sampler=test_sampler,
    #     n_support=k_shot,
    #     device=device
    # )

    # print(f"TEST mAP@1: {map1*100:.2f}% | TEST mAP@5: {map5*100:.2f}%")


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


    





    
